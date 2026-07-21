"""Tiny tagger with a shared trunk and quantization-aware prediction heads.

The trunk is sparse embedding row-sum, a depthwise temporal convolution with
neighbour injection, a bidirectional gated linear recurrence (a sequential
scan), masked global context pooling, and a gated head. Production predicts token
roles, clause boundaries, schedule kinds, speech acts, and token actionability.
Feature dimensions come from the shard manifest. Optional heads and factorized
features support existing artifact decoding; the sole trainer enables all five
production heads.

QAT is symmetric per-tensor integer quantization with a straight-through
estimator over 2-D parameters. The forward graph is identical apart from the
quantization, so selection can run with `qat=True` and track the deployed model.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SCHEDULE_KIND_NAMES = (
    "none",
    "interval",
    "daily",
    "weekly",
    "monthly",
    "ordinal",
    "bounded",
    "exception",
    "ambiguous",
)

# Must match packages/cron/src/model/speech-acts.ts. Never reorder.
SPEECH_ACT_NAMES = (
    "none",
    "request",
    "descriptive",
    "question",
    "quoted",
    "correction",
    "one-off",
    "cancel",
    "incomplete",
)

# 3-way decision the decoder actually branches on. Corpus labels collapse into this.
DECISION_ACT_NAMES = ("request", "correction", "other")
# Index in SPEECH_ACT_NAMES -> decision id.
DECISION_FROM_SPEECH = (2, 0, 2, 2, 2, 1, 2, 2, 2)

# Must match packages/cron/src/model/tokenizer.ts row ranges.
WORD_HASH_ROW0 = 140
WORD_HASH_COUNT = 256
CONSONANT_HASH_ROW0 = 396
CONSONANT_HASH_COUNT = 128
NUMBER_BUCKET_ROW0 = 564
NUMBER_BUCKET_COUNT = 16
WORD_SLOT = 4
CONSONANT_SLOT = 5
NUMBER_SLOT = 8
LEXICAL_BUCKETS = 4096
LEXICON_CLASSES = 12


def fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor int-`bits` quantization with a straight-through pass.

    `scale` maps the largest magnitude onto the positive integer ceiling
    `(1 << (bits - 1)) - 1`. The rounding residual is `detach()`ed and added
    back, so gradients flow to the full-precision weight unchanged.
    """
    ceiling = (1 << (bits - 1)) - 1
    scale = weight.detach().abs().amax()
    scale = torch.where(scale > 0, scale / ceiling, torch.ones_like(scale))
    quantized = (weight / scale).round().clamp(-ceiling, ceiling) * scale
    return weight + (quantized - weight).detach()


class QLinear(nn.Module):
    """A linear layer whose 2-D weight is fake-quantized when `qat` is set."""

    def __init__(self, in_features: int, out_features: int, bits: int = 6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.qat = False
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize(self.weight, self.bits) if self.qat else self.weight
        return F.linear(inputs, weight, self.bias)


class StatefulRecurrence(nn.Module):
    """Gated scan whose proposal and gate both read the previous state.

    `h` carries a running mix, and the gate can stay open or shut because of
    what it has already seen. Padding freezes the state, same as the input-only
    scan.
    """

    def __init__(self, hidden: int, bits: int = 6):
        super().__init__()
        self.proposal = QLinear(hidden * 2, hidden, bits)
        self.gate = QLinear(hidden * 2, hidden, bits)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        state = inputs.new_zeros(inputs.shape[0], inputs.shape[2])
        mask = valid.to(inputs.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            step_mask = mask[:, step].unsqueeze(-1)
            mixed = torch.cat((inputs[:, step], state), dim=-1)
            proposal = torch.tanh(self.proposal(mixed))
            gate = torch.sigmoid(self.gate(mixed))
            candidate = (1.0 - gate) * state + gate * proposal
            state = state + step_mask * (candidate - state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class GatedRecurrence(nn.Module):
    """One direction of a gated linear recurrence, evaluated as a sequential scan.

    At each step `h = (1 - g) * h_prev + g * proposal`, gated by a sigmoid. The
    validity mask freezes the state on padded positions, so the state and every
    later output are independent of how far the sequence was padded.
    """

    def __init__(self, hidden: int, bits: int = 6):
        super().__init__()
        self.proposal = QLinear(hidden, hidden, bits)
        self.gate = QLinear(hidden, hidden, bits)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        proposal = torch.tanh(self.proposal(inputs))
        gate = torch.sigmoid(self.gate(inputs))
        state = inputs.new_zeros(inputs.shape[0], inputs.shape[2])
        mask = valid.to(inputs.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            step_mask = mask[:, step].unsqueeze(-1)
            candidate = (1.0 - gate[:, step]) * state + gate[:, step] * proposal[:, step]
            state = state + step_mask * (candidate - state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class Tagger(nn.Module):
    """The shared trunk and its role and boundary heads."""

    def __init__(
        self,
        feature_rows: int,
        rows_per_token: int,
        role_classes: int,
        schedule_kind_classes: int = 0,
        speech_act_classes: int = 0,
        actionable_head: bool = False,
        hidden: int = 32,
        kernel_size: int = 3,
        quantization_bits: int = 6,
        factorized: bool = False,
        lexical_buckets: int = WORD_HASH_COUNT,
        lexicon_classes: int = 0,
    ):
        super().__init__()
        self.feature_rows = feature_rows
        self.rows_per_token = rows_per_token
        self.role_classes = role_classes
        self.schedule_kind_classes = schedule_kind_classes
        self.speech_act_classes = speech_act_classes
        self.actionable_head_enabled = bool(actionable_head)
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.quantization_bits = quantization_bits
        self.factorized = bool(factorized)
        self.lexical_buckets = lexical_buckets if factorized else 0
        self.lexicon_classes = lexicon_classes if factorized else 0
        self.last_pooled: torch.Tensor | None = None

        self.embedding = nn.Embedding(feature_rows + 1, hidden, padding_idx=feature_rows)
        conv_groups = 1 if factorized else hidden
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size,
            padding=kernel_size // 2,
            groups=conv_groups,
        )
        self.neighbor = QLinear(hidden, hidden, quantization_bits)
        recurrence = StatefulRecurrence if factorized else GatedRecurrence
        self.forward_recurrence = recurrence(hidden, quantization_bits)
        self.backward_recurrence = recurrence(hidden, quantization_bits)
        if factorized:
            self.word_lexical = nn.Embedding(lexical_buckets, hidden)
            self.consonant_lexical = nn.Embedding(CONSONANT_HASH_COUNT, hidden)
            self.number_lexical = nn.Embedding(NUMBER_BUCKET_COUNT, hidden)
            if lexicon_classes > 0:
                self.lexicon_class = nn.Embedding(lexicon_classes, hidden)
        self.combine = QLinear(2 * hidden, hidden, quantization_bits)
        self.context_norm = nn.LayerNorm(hidden)
        self.head_gate = QLinear(2 * hidden, hidden, quantization_bits)
        self.role_head = QLinear(hidden, role_classes, quantization_bits)
        self.boundary_head = QLinear(hidden, 1, quantization_bits)
        self.schedule_kind_head = (
            QLinear(hidden, schedule_kind_classes, quantization_bits)
            if schedule_kind_classes > 0
            else None
        )
        self.speech_act_head = (
            QLinear(hidden, speech_act_classes, quantization_bits)
            if speech_act_classes > 0
            else None
        )
        self.actionable_head = (
            QLinear(hidden, 1, quantization_bits) if actionable_head else None
        )
        self._qat = False

    @property
    def qat(self) -> bool:
        return self._qat

    @qat.setter
    def qat(self, value: bool) -> None:
        self._qat = bool(value)
        for module in self.modules():
            if isinstance(module, QLinear):
                module.qat = self._qat

    def _gather_neighbors(self, states: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """Sum the trunk state at each token's previous/next non-space neighbour."""
        batch, width, slots = neighbors.shape
        hidden = states.shape[-1]
        index = (
            neighbors.clamp(min=0)
            .reshape(batch, width * slots)
            .unsqueeze(-1)
            .expand(batch, width * slots, hidden)
        )
        gathered = torch.gather(states, 1, index).reshape(batch, width, slots, hidden)
        present = (neighbors >= 0).unsqueeze(-1).to(states.dtype)
        return (gathered * present).sum(dim=2)

    def _dedicated(
        self,
        table: nn.Embedding,
        index: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        weight = table.weight
        if self.qat:
            weight = fake_quantize(weight, self.quantization_bits)
        looked = F.embedding(index, weight)
        return looked * present.unsqueeze(-1)

    def _add_identity(
        self,
        rows: torch.Tensor,
        states: torch.Tensor,
        shared: torch.Tensor,
        lexical_ids: torch.Tensor | None,
        lexicon_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Remove identity rows from the shared sum and add full-width vectors."""

        def peel(slot: int, start: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
            row = rows[:, :, slot]
            present = (row >= start) & (row < start + count)
            looked = F.embedding(
                row.clamp(0, self.feature_rows), shared, padding_idx=self.feature_rows
            )
            return present, looked

        word_present, word_shared = peel(WORD_SLOT, WORD_HASH_ROW0, WORD_HASH_COUNT)
        consonant_present, consonant_shared = peel(
            CONSONANT_SLOT, CONSONANT_HASH_ROW0, CONSONANT_HASH_COUNT
        )
        number_present, number_shared = peel(
            NUMBER_SLOT, NUMBER_BUCKET_ROW0, NUMBER_BUCKET_COUNT
        )
        states = states - word_shared * word_present.unsqueeze(-1)
        states = states - consonant_shared * consonant_present.unsqueeze(-1)
        states = states - number_shared * number_present.unsqueeze(-1)

        if lexical_ids is None:
            word_index = (rows[:, :, WORD_SLOT] - WORD_HASH_ROW0).clamp(0, WORD_HASH_COUNT - 1)
            states = states + self._dedicated(self.word_lexical, word_index, word_present)
        else:
            buckets = self.word_lexical.num_embeddings
            states = states + self._dedicated(
                self.word_lexical,
                lexical_ids.clamp(0, buckets - 1),
                torch.ones_like(word_present),
            )
        consonant_index = (rows[:, :, CONSONANT_SLOT] - CONSONANT_HASH_ROW0).clamp(
            0, CONSONANT_HASH_COUNT - 1
        )
        number_index = (rows[:, :, NUMBER_SLOT] - NUMBER_BUCKET_ROW0).clamp(
            0, NUMBER_BUCKET_COUNT - 1
        )
        states = states + self._dedicated(
            self.consonant_lexical, consonant_index, consonant_present
        )
        states = states + self._dedicated(self.number_lexical, number_index, number_present)
        if lexicon_ids is not None and self.lexicon_classes > 0:
            present = torch.ones_like(word_present)
            states = states + self._dedicated(
                self.lexicon_class,
                lexicon_ids.clamp(0, self.lexicon_classes - 1),
                present,
            )
        return states

    def _carrier_pool(
        self, combined: torch.Tensor, roles: torch.Tensor, rows: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Pool non-space tokens by detached P(O), so the schedule span does not vote."""
        kind = rows[:, :, 0]
        not_space = valid & (kind != 3)
        outside = torch.softmax(roles.detach(), dim=-1)[..., 0]
        weights = not_space.to(combined.dtype) * outside
        denom = weights.sum(dim=1, keepdim=True)
        fallback = not_space.to(combined.dtype)
        weights = torch.where(denom < 1e-4, fallback, weights)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (combined * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        rows: torch.Tensor,
        valid: torch.Tensor,
        neighbors: torch.Tensor,
        lexical_ids: torch.Tensor | None = None,
        lexicon_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, width, _ = rows.shape
        embedding_weight = self.embedding.weight
        if self.qat:
            embedding_weight = fake_quantize(embedding_weight, self.quantization_bits)
        # Each token owns a fixed bag of sparse feature-row ids. embedding_bag
        # computes their sum directly instead of materializing the former
        # [batch, width, rows_per_token, hidden] tensor (32x larger than states).
        states = F.embedding_bag(
            rows.reshape(batch * width, self.rows_per_token),
            embedding_weight,
            mode="sum",
            padding_idx=self.feature_rows,
        ).reshape(batch, width, self.hidden)
        if self.factorized:
            states = self._add_identity(
                rows, states, embedding_weight, lexical_ids, lexicon_ids
            )
        states = states + self.neighbor(self._gather_neighbors(states, neighbors))

        mask = valid.unsqueeze(-1).to(states.dtype)
        states = states * mask
        convolved = self.depthwise(states.transpose(1, 2)).transpose(1, 2)
        states = (states + convolved) * mask

        forward = self.forward_recurrence(states, valid)
        reversed_states = torch.flip(states, dims=[1])
        reversed_valid = torch.flip(valid, dims=[1])
        backward = torch.flip(
            self.backward_recurrence(reversed_states, reversed_valid), dims=[1]
        )

        combined = self.combine(torch.cat([forward, backward], dim=-1))
        combined = self.context_norm(combined)

        denominator = mask.sum(dim=1).clamp(min=1.0)
        pooled = (combined * mask).sum(dim=1) / denominator
        pooled = pooled.unsqueeze(1).expand(-1, width, -1)
        gate = torch.sigmoid(self.head_gate(torch.cat([combined, pooled], dim=-1)))
        gated = gate * combined

        self.last_pooled = pooled[:, 0, :]
        roles = self.role_head(gated)
        boundary = self.boundary_head(gated).squeeze(-1)
        schedule_kind = (
            self.schedule_kind_head(pooled[:, 0, :])
            if self.schedule_kind_head is not None
            else pooled.new_empty((batch, 0))
        )
        if self.speech_act_head is None:
            speech_context = pooled[:, 0, :]
        elif self.factorized:
            speech_context = self._carrier_pool(combined, roles, rows, valid)
        else:
            speech_context = pooled[:, 0, :]
        speech_act = (
            self.speech_act_head(speech_context)
            if self.speech_act_head is not None
            else pooled.new_empty((batch, 0))
        )
        actionable = (
            self.actionable_head(gated).squeeze(-1)
            if self.actionable_head is not None
            else gated.new_empty((batch, 0))
        )
        return roles, boundary, schedule_kind, speech_act, actionable


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}

"""Direct checks for the tagger: padding neutrality and QAT finite gradients.

Padding neutrality is the property that makes length bucketing safe: appending
unused rows must not change a single logged probability for the real tokens.
Finite gradients under QAT prove the straight-through estimator keeps training.
Run with `python training/cron/test_model.py`.
"""
from __future__ import annotations

import torch

from dataset import SCHEDULE_KIND_NAMES as DATASET_SCHEDULE_KIND_NAMES
from dataset import schedule_kind_for_frame
from model import SCHEDULE_KIND_NAMES, Tagger

FEATURE_ROWS = 772
ROWS_PER_TOKEN = 32
ROLE_CLASSES = 15
HIDDEN = 16


def make_batch(content_length: int, padding_length: int, seed: int = 0):
    """A single sequence padded with `FEATURE_ROWS` rows and masked `False`."""
    generator = torch.Generator().manual_seed(seed)
    width = content_length + padding_length

    rows = torch.full((1, width, ROWS_PER_TOKEN), FEATURE_ROWS, dtype=torch.long)
    content = torch.randint(
        0, FEATURE_ROWS, (content_length, ROWS_PER_TOKEN), generator=generator
    )
    rows[0, :content_length] = content

    valid = torch.zeros(1, width, dtype=torch.bool)
    valid[0, :content_length] = True

    neighbors = torch.full((1, width, 2), -1, dtype=torch.long)
    for position in range(content_length):
        if position > 0:
            neighbors[0, position, 0] = position - 1
        if position + 1 < content_length:
            neighbors[0, position, 1] = position + 1
    return rows, valid, neighbors


def build_model() -> Tagger:
    torch.manual_seed(0)
    return Tagger(
        feature_rows=FEATURE_ROWS,
        rows_per_token=ROWS_PER_TOKEN,
        role_classes=ROLE_CLASSES,
        schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
        hidden=HIDDEN,
    )


def test_padding_neutrality() -> None:
    model = build_model().eval()
    content = 6
    base = make_batch(content, 0, seed=7)
    padded = make_batch(content, 9, seed=7)
    assert base[0].shape[1] != padded[0].shape[1]

    for qat in (False, True):
        model.qat = qat
        with torch.no_grad():
            base_roles, base_boundary, base_kind = model(*base)
            padded_roles, padded_boundary, padded_kind = model(*padded)
        assert torch.allclose(
            base_roles[0, :content],
            padded_roles[0, :content],
            atol=1e-5,
            rtol=1e-4,
        ), f"role logits changed under padding (qat={qat})"
        assert torch.allclose(
            base_boundary[0, :content],
            padded_boundary[0, :content],
            atol=1e-5,
            rtol=1e-4,
        ), f"boundary logits changed under padding (qat={qat})"
        assert torch.allclose(base_kind, padded_kind, atol=1e-5, rtol=1e-4), (
            f"schedule-kind logits changed under padding (qat={qat})"
        )
    print(f"padding neutrality: ok (content={content}, padded width {padded[0].shape[1]})")


def test_finite_gradients() -> None:
    model = build_model().train()
    model.qat = True
    batch = make_batch(8, 5, seed=11)
    roles, boundary, schedule_kind = model(*batch)

    target = torch.randint(0, ROLE_CLASSES, roles.shape[:-1])
    role_loss = torch.nn.functional.cross_entropy(
        roles.reshape(-1, ROLE_CLASSES), target.reshape(-1)
    )
    boundary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        boundary, torch.zeros_like(boundary)
    )
    kind_target = torch.randint(0, schedule_kind.shape[-1], schedule_kind.shape[:-1])
    kind_loss = torch.nn.functional.cross_entropy(schedule_kind, kind_target)
    (role_loss + boundary_loss + kind_loss).backward()

    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        checked += 1
    assert checked > 0, "no parameter received a gradient"
    print(f"finite qat gradients: ok ({checked} tensors)")


def test_schedule_kind_taxonomy() -> None:
    assert DATASET_SCHEDULE_KIND_NAMES == SCHEDULE_KIND_NAMES
    expected = {
        "n-interval/every-n": "interval",
        "time-of-day/digits": "daily",
        "weekday-range/through": "weekly",
        "day-of-month/on-the": "monthly",
        "ordinal-weekday/the-ordinal": "ordinal",
        "bounds/from-to": "bounded",
        "exception-days/except": "exception",
        "negative/ordinary/0": "none",
        "negative/ambiguous/0": "ambiguous",
    }
    for frame, name in expected.items():
        assert schedule_kind_for_frame(frame) == SCHEDULE_KIND_NAMES.index(name)
    print("schedule-kind taxonomy: ok")


if __name__ == "__main__":
    test_padding_neutrality()
    test_finite_gradients()
    test_schedule_kind_taxonomy()
    print("test_model: all checks passed")

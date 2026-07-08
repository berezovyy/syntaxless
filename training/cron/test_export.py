"""Wire-decode round-trip test for the int6 exporter.

The artifact that ships is the decoded one, so the strongest available check is
that decoding the packed `q` string reproduces, bit-for-bit where possible, the
quantized parameters and the logits of the same quantized checkpoint. This
catches offset, scale, and alphabet bugs without needing a trained run.

Run with `python training/cron/test_export.py`.
"""
from __future__ import annotations

import torch

from export import (
    ALPHABET,
    SYMBOL_BIAS,
    build_artifact,
    build_decoded_model,
    decode_artifact,
    pack_state,
    quantize_parameter,
)
from model import SCHEDULE_KIND_NAMES, Tagger

FEATURE_ROWS = 40
ROWS_PER_TOKEN = 5
ROLE_CLASSES = 4
HIDDEN = 8
ATOL = 1e-4


def build_model(schedule_kind_classes: int = len(SCHEDULE_KIND_NAMES)) -> Tagger:
    torch.manual_seed(0)
    return Tagger(
        feature_rows=FEATURE_ROWS,
        rows_per_token=ROWS_PER_TOKEN,
        role_classes=ROLE_CLASSES,
        schedule_kind_classes=schedule_kind_classes,
        hidden=HIDDEN,
        quantization_bits=6,
    ).eval()


def make_batch(length: int = 9):
    torch.manual_seed(11)
    rows = torch.randint(0, FEATURE_ROWS, (1, length, ROWS_PER_TOKEN))
    valid = torch.ones(1, length, dtype=torch.bool)
    neighbors = torch.full((1, length, 2), -1, dtype=torch.long)
    for position in range(length):
        if position > 0:
            neighbors[0, position, 0] = position - 1
        if position + 1 < length:
            neighbors[0, position, 1] = position + 1
    return rows, valid, neighbors


def quantized_reference(model: Tagger) -> dict[str, torch.Tensor]:
    reference: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        _, dequantized, _ = quantize_parameter(parameter)
        reference[name] = dequantized
    return reference


def test_pack_tiles_the_payload() -> None:
    model = build_model()
    segments, q = pack_state(model.state_dict())
    covered = 0
    for segment in segments:
        assert segment["offset"] == covered, "segment offsets must be contiguous"
        product = 1
        for dimension in segment["shape"]:
            product *= dimension
        assert segment["length"] == product, "segment length must equal its shape"
        covered += segment["length"]
    assert covered == len(q), "segments must tile the whole payload"
    assert set(q) <= set(ALPHABET), "payload must use only alphabet symbols"
    print(f"payload tile: ok ({len(segments)} segments, {len(q)} symbols)")


def test_symbols_stay_in_int6_range() -> None:
    model = build_model()
    segments, q = pack_state(model.state_dict())
    for segment in segments:
        chunk = q[segment["offset"] : segment["offset"] + segment["length"]]
        for symbol in chunk:
            value = ALPHABET.index(symbol) - SYMBOL_BIAS
            assert -31 <= value <= 31, f"symbol {symbol!r} out of int6 range"
    print("int6 range: ok")


def test_round_trip_reproduces_quantized_parameters() -> None:
    model = build_model()
    reference = quantized_reference(model)
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
        "scheduleKindClasses": len(SCHEDULE_KIND_NAMES),
        "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)
    decoded = decode_artifact(artifact)
    assert set(decoded) == set(reference)
    for name, expected in reference.items():
        assert torch.equal(decoded[name], expected), f"parameter mismatch: {name}"
    print(f"parameter round-trip: ok ({len(decoded)} tensors)")


def test_round_trip_reproduces_reference_logits() -> None:
    model = build_model()
    reference = quantized_reference(model)
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
        "scheduleKindClasses": len(SCHEDULE_KIND_NAMES),
        "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)

    reference_model = build_model()
    reference_model.load_state_dict(reference)
    reference_model.qat = False
    reference_model.eval()
    decoded_model = build_decoded_model(artifact)

    batch = make_batch()
    with torch.no_grad():
        reference_roles, reference_boundary, reference_kind = reference_model(*batch)
        decoded_roles, decoded_boundary, decoded_kind = decoded_model(*batch)

    role_error = float((reference_roles - decoded_roles).abs().max())
    boundary_error = float((reference_boundary - decoded_boundary).abs().max())
    kind_error = float((reference_kind - decoded_kind).abs().max())
    assert role_error <= ATOL, f"role logit error {role_error} exceeds {ATOL}"
    assert boundary_error <= ATOL, f"boundary logit error {boundary_error} exceeds {ATOL}"
    assert kind_error <= ATOL, f"schedule-kind logit error {kind_error} exceeds {ATOL}"
    print(
        "logit round-trip: ok "
        f"(role {role_error:.2e}, boundary {boundary_error:.2e}, kind {kind_error:.2e})"
    )


def test_legacy_artifact_without_schedule_kind_head() -> None:
    model = build_model(schedule_kind_classes=0)
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)
    decoded = build_decoded_model(artifact)
    with torch.no_grad():
        _, _, schedule_kind = decoded(*make_batch())
    assert schedule_kind.numel() == 0
    print("legacy artifact without schedule-kind head: ok")


def test_zero_tensor_gets_a_finite_scale() -> None:
    codes, dequantized, scale = quantize_parameter(torch.zeros(3, 4))
    assert torch.count_nonzero(codes) == 0
    assert torch.count_nonzero(dequantized) == 0
    assert scale > 0
    print("zero tensor: ok")


if __name__ == "__main__":
    test_pack_tiles_the_payload()
    test_symbols_stay_in_int6_range()
    test_round_trip_reproduces_quantized_parameters()
    test_round_trip_reproduces_reference_logits()
    test_legacy_artifact_without_schedule_kind_head()
    test_zero_tensor_gets_a_finite_scale()
    print("test_export: all checks passed")

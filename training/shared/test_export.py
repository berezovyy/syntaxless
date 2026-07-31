"""Wire-decode round-trip test for the int6 exporter.

The artifact that ships is the decoded one, so the strongest available check is
that decoding the packed `q` string reproduces, bit-for-bit where possible, the
quantized parameters and the logits of the same quantized checkpoint. This
catches offset, scale, and alphabet bugs without needing a trained run.

Run with `python training/shared/test_export.py` in the training environment.
"""
from __future__ import annotations

import torch

from artifact import (
    build_artifact,
    build_decoded_model,
    decode_artifact,
    load_labels,
    pack_state,
    quantize_parameter,
)
from model import Tagger

FEATURE_ROWS = 40
ROWS_PER_TOKEN = 5
ROLE_CLASSES = 4
HIDDEN = 8
ATOL = 1e-4


def build_model() -> Tagger:
    torch.manual_seed(0)
    return Tagger(
        feature_rows=FEATURE_ROWS,
        rows_per_token=ROWS_PER_TOKEN,
        role_classes=ROLE_CLASSES,
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


def test_round_trip_reproduces_quantized_parameters() -> None:
    model = build_model()
    reference = quantized_reference(model)
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
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
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)

    reference_model = build_model()
    reference_model.load_state_dict(reference)
    reference_model.qat = False
    reference_model.eval()
    decoded_model = build_decoded_model(artifact)

    batch = make_batch()
    with torch.no_grad():
        reference_roles, reference_boundary = reference_model(*batch)
        decoded_roles, decoded_boundary = decoded_model(*batch)

    role_error = float((reference_roles - decoded_roles).abs().max())
    boundary_error = float((reference_boundary - decoded_boundary).abs().max())
    assert role_error <= ATOL, f"role logit error {role_error} exceeds {ATOL}"
    assert boundary_error <= ATOL, f"boundary logit error {boundary_error} exceeds {ATOL}"
    print(
        "logit round-trip: ok "
        f"(role {role_error:.2e}, boundary {boundary_error:.2e})"
    )


def test_artifact_has_no_schedule_kind_head() -> None:
    segments, q = pack_state(build_model().state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)
    assert "scheduleKindClasses" not in artifact
    assert len(artifact["segments"]) == 23, "runtime expects exactly 23 tensors"


def test_labels_match_each_package_vocabulary() -> None:
    for package in ("validation", "filter", "sort", "retry", "retention", "format"):
        labels = load_labels(package)
        assert labels[0] == "O", package
        assert len(set(labels)) == len(labels), package


if __name__ == "__main__":
    test_round_trip_reproduces_quantized_parameters()
    test_round_trip_reproduces_reference_logits()
    test_artifact_has_no_schedule_kind_head()
    test_labels_match_each_package_vocabulary()
    print("test_export: all checks passed")

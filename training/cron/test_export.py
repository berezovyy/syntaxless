"""Wire-decode round-trip test for the int6 exporter.

The artifact that ships is the decoded one, so the strongest available check is
that decoding the packed `q` string reproduces, bit-for-bit where possible, the
quantized parameters and the logits of the same quantized checkpoint. This
catches offset, scale, and alphabet bugs without needing a trained run.

Run with `python training/cron/test_export.py`.
"""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from dataset import ROLE_LABELS, Dataset, repo_root
from export import (
    artifact_sha256,
    calibrated_checkpoint,
    build_artifact,
    build_decoded_model,
    decode_artifact,
    pack_state,
    quantize_parameter,
    write_candidate_module,
)
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger
from test_dataset import _write_tiny_shard

FEATURE_ROWS = 40
ROWS_PER_TOKEN = 5
ROLE_CLASSES = 4
HIDDEN = 8
ATOL = 1e-4


def build_model(
    schedule_kind_classes: int = len(SCHEDULE_KIND_NAMES),
    speech_act_classes: int = len(SPEECH_ACT_NAMES),
    actionable_head: bool = True,
) -> Tagger:
    torch.manual_seed(0)
    return Tagger(
        feature_rows=FEATURE_ROWS,
        rows_per_token=ROWS_PER_TOKEN,
        role_classes=ROLE_CLASSES,
        schedule_kind_classes=schedule_kind_classes,
        speech_act_classes=speech_act_classes,
        actionable_head=actionable_head,
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


def test_artifact_identity_matches_typescript_for_complete_calibrated_weights() -> None:
    model = build_model()
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
    }
    artifact = build_artifact(
        segments, q, ["O", "RECUR", "GLUE", "EDGE"], checkpoint, HIDDEN, 0.0
    )
    artifact.update({
        "energyThreshold": -1e-7,
        "mahalanobisMean": [1e20, -0.0, 1.23456789012345, 0.0],
        "mahalanobisVariance": [1e-8, 1.0, 3.0, 1e20],
        "mahalanobisThreshold": 1.0,
    })
    expected = artifact_sha256(artifact)
    # Input arrives through JSON just as a generated weight module does. This
    # checks the real consumer rather than duplicating the encoder in a test.
    result = subprocess.run(
        [
            "bun", "--eval",
            'import { artifactSha256 } from "./packages/cron/training/src/weights-hash.ts";'
            ' process.stdout.write(artifactSha256(JSON.parse(await Bun.stdin.text())));',
        ],
        input=json.dumps(artifact), text=True, capture_output=True,
        cwd=repo_root(), check=True,
    )
    assert result.stdout == expected, "export and evaluation must identify the same artifact"
    assert artifact_sha256(dict(reversed(list(artifact.items())))) == expected

    for field, value in (
        ("energyThreshold", -2e-7),
        ("boundaryThreshold", 0.5),
        ("labels", ["O", "GLUE", "RECUR", "EDGE"]),
    ):
        changed = {**artifact, field: value}
        assert changed["q"] == artifact["q"]
        assert artifact_sha256(changed) != expected, f"identity ignores {field}"
    changed = copy.deepcopy(artifact)
    changed["segments"][0]["scale"] *= 2
    assert changed["q"] == artifact["q"]
    assert artifact_sha256(changed) != expected, "identity ignores tensor scale"


def test_candidate_report_matches_evaluation_after_calibration() -> None:
    for role_classes in (77, len(ROLE_LABELS)):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            _write_tiny_shard(directory, name="corpus-validation")
            manifest_path = directory / "corpus-validation.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["roleClasses"] = role_classes
            if role_classes == 77:
                manifest["labelCounts"].pop("CONTEXT_NUMBER", None)
            manifest_path.write_text(json.dumps(manifest))
            model = Tagger(
                feature_rows=16, rows_per_token=2, role_classes=role_classes,
                hidden=HIDDEN, speech_act_classes=len(SPEECH_ACT_NAMES),
                schedule_kind_classes=len(SCHEDULE_KIND_NAMES), actionable_head=True,
            ).eval()
            checkpoint = directory / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "config": {"data_prefix": str(directory / "corpus")},
                "roleClasses": role_classes, "featureRows": 16, "rowsPerToken": 2,
                "speechActClasses": len(SPEECH_ACT_NAMES),
                "speechActLabels": list(SPEECH_ACT_NAMES),
                "scheduleKindClasses": len(SCHEDULE_KIND_NAMES),
                "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
            }, checkpoint)
            validation = Dataset(directory / "corpus-validation")
            selected_artifact, _, _, _ = calibrated_checkpoint(checkpoint, validation)
            selected_sha = artifact_sha256(selected_artifact)
            candidate = write_candidate_module(
                checkpoint,
                directory / "weights.gen.ts",
                validation,
                expected_artifact_sha256=selected_sha,
            )
            report = json.loads((directory / "export.json").read_text())
            result = subprocess.run(
                [
                    "bun", "--eval",
                    'import { loadEngine } from "./packages/cron/training/src/evaluate.ts";'
                    ' const engine = await loadEngine("no-dist", await Bun.stdin.text());'
                    ' process.stdout.write(engine.artifactSha256);',
                ],
                input=str(candidate), text=True, capture_output=True,
                cwd=repo_root(), check=True,
            )
            assert result.stdout == report["artifactSha256"]
            assert f"artifactSha256: {result.stdout}" in candidate.read_text()
            assert report["calibrationSplit"] == str(directory / "corpus-validation")
            assert report["promoted"] is False
            before_module = candidate.read_bytes()
            before_report = (directory / "export.json").read_bytes()
            try:
                write_candidate_module(checkpoint, candidate, validation, expected_artifact_sha256="0" * 64)
            except ValueError as error:
                assert "used for selection" in str(error)
            else:
                raise AssertionError("An export changed after decoded checkpoint selection")
            assert candidate.read_bytes() == before_module
            assert (directory / "export.json").read_bytes() == before_report


def test_artifact_identity_rejects_non_finite_metadata() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            artifact_sha256({"energyThreshold": value})
        except ValueError as error:
            assert "Non-finite" in str(error)
        else:
            raise AssertionError("Invalid metadata cannot have a report identity")


def test_factorized_wire_inference_matches_typescript() -> None:
    """Exercise both supported lexical layouts, every head, and a padded tail."""
    fields = ("logits", "boundaryLogits", "scheduleKindLogits", "speechActLogits", "actionableLogits")
    rows = torch.full((1, 5, 32), 772, dtype=torch.long)
    rows[0, 1, 0] = 3  # Space separates the two content tokens.
    for index in (0, 2):
        rows[0, index, 0] = 0
        rows[0, index, 1] = 12 + index
        rows[0, index, 4] = 140 + index
        rows[0, index, 5] = 396 + index
        rows[0, index, 8] = 564 + index
    valid = torch.tensor([[True, True, True, False, False]])
    neighbors = torch.tensor([[[-1, 2], [0, 2], [0, -1], [2, -1], [2, -1]]])
    for lexical_buckets, lexicon_classes in ((256, 0), (4096, 12)):
        torch.manual_seed(7)
        model = Tagger(
            feature_rows=772, rows_per_token=32, role_classes=ROLE_CLASSES,
            hidden=HIDDEN, factorized=True, lexical_buckets=lexical_buckets,
            lexicon_classes=lexicon_classes,
            schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
            speech_act_classes=len(SPEECH_ACT_NAMES), actionable_head=True,
        ).eval()
        segments, q = pack_state(model.state_dict())
        artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, {
            "featureRows": 772, "rowsPerToken": 32, "roleClasses": ROLE_CLASSES,
            "scheduleKindClasses": len(SCHEDULE_KIND_NAMES),
            "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
            "speechActClasses": len(SPEECH_ACT_NAMES),
            "speechActLabels": list(SPEECH_ACT_NAMES),
        }, HIDDEN, 0.0)
        decoded = build_decoded_model(artifact)
        lexical = torch.tensor([[99, 0, 1234, 0, 0]]) if lexical_buckets == 4096 else None
        lexicon = torch.tensor([[1, 0, 8, 0, 0]]) if lexicon_classes else None
        with torch.no_grad():
            reference = decoded(rows, valid, neighbors, lexical, lexicon)
        result = subprocess.run([
            "bun", "--eval",
            'import { createInference } from "./packages/cron/src/model/inference.ts";'
            ' const input = JSON.parse(await Bun.stdin.text());'
            ' const prediction = createInference(input.artifact).inferRows(input.rows, 3, true,'
            ' input.lexical === null ? undefined : Uint16Array.from(input.lexical),'
            ' input.lexicon === null ? undefined : Uint8Array.from(input.lexicon));'
            ' const fields = ["logits", "boundaryLogits", "scheduleKindLogits",'
            ' "speechActLogits", "actionableLogits", "pooled"];'
            ' process.stdout.write(JSON.stringify(Object.fromEntries(fields.map(name =>'
            ' [name, Array.from(prediction[name] ?? [])]))));',
        ], input=json.dumps({
            "artifact": artifact, "rows": rows.reshape(-1).tolist(),
            "lexical": lexical.reshape(-1).tolist() if lexical is not None else None,
            "lexicon": lexicon.reshape(-1).tolist() if lexicon is not None else None,
        }), text=True, capture_output=True, cwd=repo_root(), check=True)
        actual = json.loads(result.stdout)
        for name, expected in zip(fields, reference, strict=True):
            torch.testing.assert_close(
                torch.tensor(actual[name]).reshape(expected.shape), expected,
                atol=ATOL, rtol=0, msg=lambda message: f"{lexical_buckets} {name}: {message}",
            )
        torch.testing.assert_close(
            torch.tensor(actual["pooled"]).reshape(decoded.last_pooled.shape),
            decoded.last_pooled, atol=ATOL, rtol=0,
        )


def test_wire_decoder_rejects_float32_overflow() -> None:
    for symbol in ("+", "A"):
        artifact = {
            "q": symbol,
            "segments": [{
                "name": "overflow", "offset": 0, "length": 1,
                "shape": [1], "scale": 1e38,
            }],
        }
        try:
            decode_artifact(artifact)
        except ValueError as error:
            assert "Float32" in str(error)
        else:
            raise AssertionError("Wire decoding accepted a non-finite Float32 tensor")
    decoded = decode_artifact({
        "q": "Agf+",
        "segments": [{
            "name": "large", "offset": 0, "length": 4,
            "shape": [4], "scale": 1e37,
        }],
    })["large"]
    assert torch.isfinite(decoded).all()
    assert decoded[0] < -1e38 and decoded[3] > 1e38
    assert decoded[1] > 0 and decoded[2] == 0


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
        "speechActClasses": len(SPEECH_ACT_NAMES),
        "speechActLabels": list(SPEECH_ACT_NAMES),
        "actionableHead": True,
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)
    decoded = decode_artifact(artifact)
    assert set(decoded) == set(reference)
    for name, expected in reference.items():
        assert torch.equal(decoded[name], expected), f"parameter mismatch: {name}"


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
        "speechActClasses": len(SPEECH_ACT_NAMES),
        "speechActLabels": list(SPEECH_ACT_NAMES),
        "actionableHead": True,
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)

    reference_model = build_model()
    reference_model.load_state_dict(reference)
    reference_model.qat = False
    reference_model.eval()
    decoded_model = build_decoded_model(artifact)

    batch = make_batch()
    with torch.no_grad():
        (
            reference_roles,
            reference_boundary,
            reference_kind,
            reference_speech,
            reference_actionable,
        ) = reference_model(*batch)
        (
            decoded_roles,
            decoded_boundary,
            decoded_kind,
            decoded_speech,
            decoded_actionable,
        ) = decoded_model(*batch)

    role_error = float((reference_roles - decoded_roles).abs().max())
    boundary_error = float((reference_boundary - decoded_boundary).abs().max())
    kind_error = float((reference_kind - decoded_kind).abs().max())
    speech_error = float((reference_speech - decoded_speech).abs().max())
    actionable_error = float((reference_actionable - decoded_actionable).abs().max())
    assert role_error <= ATOL, f"role logit error {role_error} exceeds {ATOL}"
    assert boundary_error <= ATOL, f"boundary logit error {boundary_error} exceeds {ATOL}"
    assert kind_error <= ATOL, f"schedule-kind logit error {kind_error} exceeds {ATOL}"
    assert speech_error <= ATOL, f"speech-act logit error {speech_error} exceeds {ATOL}"
    assert actionable_error <= ATOL, (
        f"actionable logit error {actionable_error} exceeds {ATOL}"
    )


def test_artifact_without_schedule_kind_head() -> None:
    model = build_model(
        schedule_kind_classes=0, speech_act_classes=0, actionable_head=False
    )
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)
    decoded = build_decoded_model(artifact)
    with torch.no_grad():
        _, _, schedule_kind, speech_act, actionable = decoded(*make_batch())
    assert schedule_kind.numel() == 0
    assert speech_act.numel() == 0
    assert actionable.numel() == 0


def test_artifact_without_discourse_heads() -> None:
    model = build_model(
        schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
        speech_act_classes=0,
        actionable_head=False,
    )
    segments, q = pack_state(model.state_dict())
    checkpoint = {
        "featureRows": FEATURE_ROWS,
        "rowsPerToken": ROWS_PER_TOKEN,
        "roleClasses": ROLE_CLASSES,
        "scheduleKindClasses": len(SCHEDULE_KIND_NAMES),
        "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
    }
    artifact = build_artifact(segments, q, ["O"] * ROLE_CLASSES, checkpoint, HIDDEN, 0.0)
    decoded = build_decoded_model(artifact)
    with torch.no_grad():
        _, _, _, speech_act, actionable = decoded(*make_batch())
    assert speech_act.numel() == 0
    assert actionable.numel() == 0


if __name__ == "__main__":
    test_round_trip_reproduces_quantized_parameters()
    test_round_trip_reproduces_reference_logits()
    test_artifact_without_schedule_kind_head()
    test_artifact_without_discourse_heads()
    print("test_export: all checks passed")

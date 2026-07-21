"""Decoded selection checks with actual TypeScript inference, not claimed scores."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import tempfile
from pathlib import Path

from dataset import repo_root
from selection import decoded_preserves, decoded_rank, selects_checkpoint, significant_count_increase, significant_rate_drop, verify_preparation


def _runtime_reports(directory: Path, *, imbalance: bool = False, qualified: bool = False) -> list[dict]:
    script = r'''
        import { join } from "node:path";
        import { createTestWeights } from "./packages/cron/training/tests/helpers/test-weights.ts";
        import { checkpointRecords } from "./packages/cron/training/tests/helpers/checkpoint-records.ts";
        import { scoreCheckpoint } from "./packages/cron/training/src/checkpoint.ts";
        import { labelId } from "./packages/cron/src/model/labels.ts";
        const {directory, imbalance, qualified} = JSON.parse(await Bun.stdin.text()), corpus = join(directory, "corpus.jsonl");
        // Fixed meanings with unequal construction support expose an aggregate
        // improvement that loses every positive in a smaller construction.
        const rows = imbalance ? [
          ...Array.from({length: 40}, (_, index) => ({...checkpointRecords[0], id: `monday-${index}`, template: "monday/bare"})),
          ...Array.from({length: 120}, (_, index) => ({...checkpointRecords[1], id: `tuesday-${index}`, template: "tuesday/bare"})),
          checkpointRecords[2],
        ] : qualified ? [checkpointRecords[0], {
          ...checkpointRecords[2], text: "!!!",
          spans: [{start: 0, end: 3, label: "O", clauseStart: false, actionable: false}],
        }] : checkpointRecords;
        await Bun.write(corpus, rows.map(row => JSON.stringify(row) + "\n").join(""));
        const reports = [];
        const roles = imbalance ? [labelId.WEEKDAY_MON, labelId.WEEKDAY_TUE, labelId.O] : [labelId.WEEKDAY_MON, labelId.O];
        for (const role of roles) {
          const module = join(directory, `weights-${role}.ts`);
          await Bun.write(module, `export const weights = ${JSON.stringify(createTestWeights(role, true))};`);
          reports.push(await scoreCheckpoint(corpus, module, "validation"));
        }
        process.stdout.write(JSON.stringify(reports));
    '''
    result = subprocess.run(["bun", "--eval", script], input=json.dumps({"directory": str(directory), "imbalance": imbalance, "qualified": qualified}), text=True,
                            capture_output=True, cwd=repo_root(), check=True)
    return json.loads(result.stdout)


def test_actual_intent_scores_rank_above_always_refusal_and_require_same_inputs() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        partial, refusing = _runtime_reports(Path(scratch))
        assert partial["positiveExact"] == 1 and partial["positiveWrongAccepted"] == 1
        assert partial["negativeFalseAccepted"] == 1
        assert refusing["positiveExact"] == 0 and refusing["positiveAccepted"] == 0
        assert decoded_rank(partial) > decoded_rank(refusing)
        assert not partial["syntheticGatePassed"] and not refusing["syntheticGatePassed"]
        assert decoded_preserves(partial, partial)
        for key in ("macroPositiveExact", "wrongAcceptedRateAcceptedPositive"):
            changed = copy.deepcopy(partial)
            changed[key] = 1
            try:
                decoded_rank(changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Claimed {key} replaced actual intent counts")
        changed = copy.deepcopy(partial)
        changed["positiveExactRate"] = 1
        try:
            decoded_rank(changed)
        except ValueError as error:
            assert "disagrees with counts" in str(error)
        else:
            raise AssertionError("Claimed rates replaced actual intent counts")
        for key in ("selectedRecordsSha256", "corpusSha256", "sourceHashes", "positiveTotal"):
            changed = copy.deepcopy(partial)
            changed[key] = {} if key == "sourceHashes" else "changed"
            try:
                decoded_preserves(changed, partial)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Checkpoint comparison accepted changed {key}")


def test_actual_mean_improvement_cannot_replace_a_checkpoint_with_a_construction_loss() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        monday, tuesday, refusing = _runtime_reports(Path(scratch), imbalance=True)
        assert monday["positiveExact"] == 40 and tuesday["positiveExact"] == 120
        assert monday["positiveWrongAccepted"] == 120 and tuesday["positiveWrongAccepted"] == 40
        assert decoded_rank(tuesday) > decoded_rank(monday)
        assert decoded_preserves(tuesday, tuesday)
        assert not decoded_preserves(tuesday, monday)
        assert monday["byKind"]["monday"]["positiveExact"] == 40
        assert tuesday["byKind"]["monday"]["positiveExact"] == 0
        assert not decoded_preserves(monday, refusing)
        assert selects_checkpoint(monday, refusing)
        assert selects_checkpoint(tuesday, monday)


def test_passing_runtime_checkpoint_cannot_be_replaced_by_a_failing_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        passing, refusing = _runtime_reports(Path(scratch), qualified=True)
        assert passing["syntheticGatePassed"] and not refusing["syntheticGatePassed"]
        assert selects_checkpoint(passing, refusing)
        assert not selects_checkpoint(refusing, passing)
        assert not selects_checkpoint(passing, passing)


def test_safety_comparison_uses_declared_noise_bounds_instead_of_only_the_mean() -> None:
    before, after, support = 0.9, 0.87, 5000
    bound = max(2 * math.sqrt((before * (1 - before) + after * (1 - after)) / support), 1 / support)
    assert before - after > bound
    assert significant_rate_drop(before, after, support)
    assert not significant_rate_drop(0.9, 0.8, 10)
    assert not significant_count_increase(0, 1)
    assert significant_count_increase(0, 5)


def test_diagnostic_inputs_cannot_authorize_parameter_training() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        _runtime_reports(directory)
        (directory / "preparation.json").write_text(json.dumps({
            "trainingEligible": False, "rows": 3,
        }))
        try:
            verify_preparation(directory / "corpus")
        except ValueError as error:
            assert "not diagnostic data" in str(error)
        else:
            raise AssertionError("A diagnostic corpus authorized training")

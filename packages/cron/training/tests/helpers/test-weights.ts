import type { WeightsArtifact } from "../../../src/model/decode";
import { LABEL_COUNT, labelId, LABELS } from "../../../src/model/labels";
import { weights as legacy } from "../../../src/model/weights.gen";

/**
 * Exercise real candidate loading and inference without a local training run.
 * Reuse the committed encoder with a deterministic typed role head. These are
 * test weights, not a trained candidate or evidence of parsing accuracy.
 */
export function createTestWeights(role = labelId.WEEKDAY_MON, fixedRequest = false) {
  const roleWeight = "f".repeat(LABEL_COUNT * legacy.hidden); // int6 zero

  const roleBias = Array.from({ length: LABEL_COUNT }, (_, id) => (id === role ? "p" : "f")).join(
    "",
  );

  let weights: WeightsArtifact = {
    ...legacy,
    roleClasses: LABEL_COUNT,
    labels: LABELS,
    q: legacy.q + roleWeight + roleBias,
    segments: [
      ...legacy.segments.filter((segment) => !segment.name.startsWith("role_head.")),
      {
        name: "role_head.weight",
        offset: legacy.q.length,
        length: roleWeight.length,
        shape: [LABEL_COUNT, legacy.hidden],
        scale: 1,
      },
      {
        name: "role_head.bias",
        offset: legacy.q.length + roleWeight.length,
        length: roleBias.length,
        shape: [LABEL_COUNT],
        scale: 1,
      },
    ],
  };

  if (fixedRequest) {
    let q = weights.q;

    const forced = new Set([
      "boundary_head",
      "actionable_head",
      "schedule_kind_head",
      "speech_act_head",
    ]);

    const segments = weights.segments.map((segment) => {
      const [head, parameter] = segment.name.split(".");

      if (!forced.has(head!)) return segment;

      const selected =
        head === "schedule_kind_head"
          ? weights.scheduleKindLabels!.indexOf("weekly")
          : head === "speech_act_head"
            ? weights.speechActLabels!.indexOf("request")
            : 0;

      const symbols = Array.from({ length: segment.length }, (_, index) =>
        parameter === "bias" && index === selected ? "p" : "f",
      ).join("");

      const offset = q.length;

      q += symbols;

      return { ...segment, offset, scale: 1 };
    });

    const { energyThreshold, mahalanobisMean, mahalanobisVariance, mahalanobisThreshold, ...base } =
      weights;

    weights = { ...base, q, segments };
  }

  return weights;
}

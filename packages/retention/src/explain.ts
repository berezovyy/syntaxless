import type { RetentionIntent } from "./types.js";
import { validate } from "./validate.js";

export function explain(intent: RetentionIntent): string {
  validate(intent);

  const parts = intent.tiers.map((tier, i) => {
    const keeps = `Tier ${i + 1} keeps the newest`;
    const perBucket = `per local ${tier.bucket}`;

    if (tier.window.kind === "calendar") {
      const periods = `${tier.window.count} calendar ${tier.window.unit} periods`;

      return `${keeps} backup ${perBucket} in ${periods}, including the current period.`;
    }

    const elapsed = `${tier.window.milliseconds} milliseconds`;

    return (
      `${keeps} eligible backup ${perBucket} within the last ${elapsed}, ` +
      "including the exact cutoff."
    );
  });

  parts.push(
    "Previews require a caller timezone and reference instant. " +
      "Weeks begin on Monday, and years begin on January 1.",
  );

  parts.push(
    "Tier selections are combined, and tied timestamps use ascending Unicode identifier order.",
  );

  parts.push("The preview returns recommendations and never deletes backups.");

  return parts.join(" ");
}

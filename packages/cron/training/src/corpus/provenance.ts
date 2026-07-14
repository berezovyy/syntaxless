import { z } from "zod";

/** Producer (`rule` / `llm`) is not authorship. Never infer human origin from it. */
export const provenanceSchema = z
  .object({
    authorship: z.enum(["ai-authored", "human-authored", "public-source", "programmatic"]),
    sourceId: z.string().min(1),
    /** Paraphrases, counterfactuals and value substitutions share this split key. */
    groupId: z.string().min(1),
    constructionId: z.string().min(1),
    sourceRecordId: z.string().min(1).optional(),
    license: z.string().min(1).optional(),
    attribution: z.string().min(1).optional(),
  })
  .strict();

export type CorpusProvenance = z.infer<typeof provenanceSchema>;

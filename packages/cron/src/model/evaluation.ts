/** Internal build entry used by the artifact promotion gate.
 *
 * This file is not part of the package export map. It exposes the exact
 * code-split tokenizer and inference functions used by the public parser so a
 * candidate weights module can be evaluated without a second implementation.
 */
export { inferTokens } from "./inference.js";
export { LABELS } from "./labels.js";
export { tokenize } from "./tokenizer.js";
export { weights } from "./weights.gen.js";

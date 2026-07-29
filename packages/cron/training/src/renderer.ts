/** Shape of one rendered training example: surface text plus aligned role spans. */
import type { SpeechAct } from "../../src/model/speech-acts";
import type { Span } from "./sentence";

export interface RenderedExample {
  text: string;
  spans: Span[];
  /** `${family}/${style}`; recorded as the example's template name. */
  template: string;
  style: string;
  speechAct: SpeechAct;
}

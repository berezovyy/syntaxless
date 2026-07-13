import { describe, expect, test } from "bun:test";

import { scopeActionableCorrection } from "./model";

describe("actionable correction scoping", () => {
  test.each([
    ["Actually, make that every 20 minutes, not every hour.", "every 20 minutes"],
    ["Sorry, I meant Tuesday at 10 am rather than Monday at nine.", "Tuesday at 10 am"],
    [
      "I said Friday by mistake. Please make it Thursday every week.",
      "Please make it Thursday every week.",
    ],
    ["Not Monday and Wednesday after all; schedule Friday at four.", "schedule Friday at four."],
    ["Move it from Tuesday at eight to Thursday at eleven.", "schedule Thursday at eleven."],
    ["Instead of noon, make the daily reminder 2 pm.", "make the daily reminder 2 pm."],
    [
      "Actually, forget the hourly check and poll every 26 minutes instead.",
      "poll every 26 minutes",
    ],
    [
      "Not Wednesday at eight—make the meeting reminder Friday at 10:30 a.m.",
      "make the meeting reminder Friday at 10:30 a.m.",
    ],
    ["I said the 12th, but I meant the 21st of every month.", "the 21st of every month."],
    [
      "Scratch the daily plan; schedule it every Monday at 7 p.m.",
      "schedule it every Monday at 7 p.m.",
    ],
    [
      "Change that from weekends to weekdays, still at 11:15.",
      "schedule weekdays, still at 11:15.",
    ],
    ["No, I need it every two hours, not every two minutes.", "every two hours"],
    [
      "Make that 8:05 tonight and every evening thereafter, instead of 8:50.",
      "make 8:05 tonight and every evening thereafter",
    ],
    [
      "I gave you Tuesday by mistake. Please schedule Thursday at 09:40.",
      "Please schedule Thursday at 09:40.",
    ],
    ["Correction: daily at 21:15 rather than weekly at 21:15.", "daily at 21:15"],
    [
      "Sorry, I meant a quarter past eight each morning, not a quarter to eight.",
      "a quarter past eight each morning",
    ],
    ["Sorry, use Saturday at 10:20 a.m. instead of Sunday at noon.", "use Saturday at 10:20 a.m."],
    [
      "I said the 9th in error. Schedule the 19th of every month.",
      "Schedule the 19th of every month.",
    ],
    ["No, I need weekdays at 07:25 rather than weekends.", "weekdays at 07:25"],
    ["Use every three hours instead—forget the three-minute setting.", "Use every three hours"],
    ["The morning time was a mistake; make it 7:15 every evening.", "make it 7:15 every evening."],
    ["Revise my request to monthly on the 26th, rather than weekly.", "monthly on the 26th"],
    ["Apologies—make it Thursday at 9:35 a.m., not Tuesday at noon.", "make Thursday at 9:35 a.m."],
    ["No—working days at 08:15, rather than Saturday and Sunday.", "working days at 08:15"],
    ["Correction — Tuesday and Saturday at 14:00, not Thursday.", "Tuesday and Saturday at 14:00"],
    [
      "Morning won’t work. Set the daily reminder for 8:25 in the evening.",
      "Set the daily reminder for 8:25 in the evening.",
    ],
    ["Please revise the cadence to monthly on day 25 instead of weekly.", "monthly on day 25"],
    ["Sorry—use Friday at 11:25 a.m., not Monday at five.", "use Friday at 11:25 a.m."],
    ["No, business days at 09:05 instead of weekends.", "business days at 09:05"],
  ])("keeps only the selected replacement in %s", (source, expected) => {
    expect(scopeActionableCorrection(source)).toBe(expected);
  });

  test.each([
    "Cancel the weekday idea.",
    "Actually, never mind.",
    "Monday or Tuesday might work; I have not decided.",
  ])("does not turn a cancellation or unresolved choice into a request", (source) => {
    expect(scopeActionableCorrection(source)).toBe(source);
  });
});

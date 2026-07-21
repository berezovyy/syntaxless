const NUMBER_ONES = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
] as const;

const NUMBER_TENS: Readonly<Record<number, string>> = {
  20: "twenty",
  30: "thirty",
  40: "forty",
  50: "fifty",
  60: "sixty",
  70: "seventy",
  80: "eighty",
  90: "ninety",
};

/** Spelled-out numbers zero to one hundred. Hyphenated and spaced compound forms are both rows. */
export const NUMBER_WORDS: Record<string, number> = (() => {
  const words: Record<string, number> = {};

  NUMBER_ONES.forEach((word, value) => {
    words[word] = value;
  });

  for (const [tens, word] of Object.entries(NUMBER_TENS)) {
    const tensValue = Number(tens);

    words[word] = tensValue;

    for (let ones = 1; ones <= 9; ones++) {
      const onesWord = NUMBER_ONES[ones]!;

      words[`${word} ${onesWord}`] = tensValue + ones;
      words[`${word}-${onesWord}`] = tensValue + ones;
    }
  }

  words.hundred = 100;
  words["one hundred"] = 100;

  return words;
})();

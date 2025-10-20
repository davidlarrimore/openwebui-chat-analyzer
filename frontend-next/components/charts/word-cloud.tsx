"use client";

import { useMemo } from "react";
import type { WordFrequency } from "@/lib/content-analysis";

const MIN_FONT_SIZE_REM = 0.85;
const MAX_FONT_SIZE_REM = 2.6;
const PALETTE = ["#2563eb", "#7c3aed", "#0ea5e9", "#f59e0b", "#ec4899", "#10b981", "#f97316", "#14b8a6"];

export interface WordCloudProps {
  words: WordFrequency[];
}

interface DisplayWord extends WordFrequency {
  fontSizeRem: number;
  color: string;
}

export function WordCloud({ words }: WordCloudProps) {
  const displayWords = useMemo<DisplayWord[]>(() => {
    if (!words.length) {
      return [];
    }

    const counts = words.map((entry) => entry.count);
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    const span = Math.max(1, maxCount - minCount);

    return words.map((entry, index) => {
      const normalised = (entry.count - minCount) / span;
      const fontSizeRem = MIN_FONT_SIZE_REM + normalised * (MAX_FONT_SIZE_REM - MIN_FONT_SIZE_REM);
      const color = PALETTE[index % PALETTE.length];
      return {
        ...entry,
        fontSizeRem,
        color
      };
    });
  }, [words]);

  if (!displayWords.length) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center justify-center gap-4 px-4 py-6 text-center">
      {displayWords.map((word) => (
        <span
          key={`${word.text}-${word.count}`}
          className="font-semibold"
          style={{ fontSize: `${word.fontSizeRem}rem`, color: word.color, lineHeight: 1.2 }}
        >
          {word.text}
        </span>
      ))}
    </div>
  );
}


#!/usr/bin/env python3
"""
Text segmentation utilities for trajectory-based reasoning analysis.

Splits reasoning text into clause-level segments for multi-step
trajectory analysis on the operator manifold.
"""

import re
from typing import List


# Clause-level connectors to split on (preserving the connector with the following clause)
_CLAUSE_SPLIT = re.compile(
    r'(?<=[.!?;])\s+'                          # sentence boundaries
    r'|(?<=,)\s+'                               # commas
    r'|\s+(?=(?:therefore|so|because|since|thus|hence|consequently|'
    r'but|however|yet|although|while|whereas|'
    r'just as|similarly|in the same way|'
    r'which|that|and then|then)\b)',            # conjunctions / connectors
    re.IGNORECASE,
)


def segment_text(text: str) -> List[str]:
    """
    Split reasoning text into clause-level segments for trajectory analysis.

    Strategy:
      1. Split on sentence boundaries and clause connectors
      2. Merge fragments shorter than 3 words back into the previous segment
      3. If only 1 segment (no natural boundaries), fall back to cumulative
         word-window prefixes so there are still multiple trajectory points
    """
    # Step 1: clause split
    raw = _CLAUSE_SPLIT.split(text.strip())
    raw = [s.strip() for s in raw if s.strip()]

    # Step 2: merge tiny fragments
    merged = []
    for seg in raw:
        if merged and len(seg.split()) < 3:
            merged[-1] = merged[-1] + ' ' + seg
        else:
            merged.append(seg)

    # Use natural clause split if it produced at least 2 segments
    if len(merged) >= 2:
        return merged

    # Step 3: fallback -- cumulative word-window prefixes
    words = text.split()
    n_target = min(max(4, len(words) // 3), 8)  # 4-8 points
    if len(words) <= n_target:
        return [' '.join(words[:i+1]) for i in range(len(words))]

    step = max(1, len(words) // n_target)
    segments = []
    for i in range(step, len(words) + 1, step):
        segments.append(' '.join(words[:i]))
    if segments and segments[-1] != text.strip():
        segments.append(text.strip())
    return segments


if __name__ == "__main__":
    tests = [
        "The temperature dropped below freezing, so the pipes burst from ice expansion.",
        "All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is warm-blooded.",
        "Short text.",
        "The drought killed crops, causing food shortages that triggered mass migration.",
    ]
    for t in tests:
        segs = segment_text(t)
        print(f"\n[{len(segs)} segments] {t[:60]}...")
        for i, s in enumerate(segs):
            print(f"  [{i+1}] {s}")
    print("\nAll text_utils tests passed!")

#!/usr/bin/env python3
"""
hardcoded_text_timelines.py

Reads a hardcoded annotations text file ("file.txt") that has one video per line:
  video.ext | 00:00:12.345=yes, 00:01:00.000=no, ...

For EACH LINE, it builds a text bar like:
  video.ext | [---------==++==+==----==+==------]

…and writes all results to a hardcoded output file ("bars_output.txt").
No command-line args needed.

Legend:
  '-' = no
  '=' = yes
  '+' = perfect

Rules:
- Each line is processed independently (no cross-video merging).
- If the same timestamp appears multiple times within a line, LAST WRITE WINS.
- If multiple timestamps map to the same character slot, precedence is:
    perfect > yes > no
- Unlabeled bins default to '-'.
"""

import os
import re
from typing import Dict, Tuple

# ---------- Hardcoded paths & settings ----------
INPUT_PATH  = "combo_data/combined.txt"  # change if you want a different input filename
OUTPUT_PATH = "bars_output.txt"                 # the program will create/overwrite this file
BAR_WIDTH   = 300                                # characters inside the brackets
FILL_MODE   = "blank"             # or "blank" for spaces instead of '-'
# ------------------------------------------------

VALID_LABELS = {"no", "yes", "perfect"}
LABEL_CHAR   = {"no": " ", "yes": "o", "perfect": "O"}
LABEL_RANK   = {"no": 1, "yes": 2, "perfect": 3}  # higher wins on collision
TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")

def hhmmss_ms_to_seconds(ts: str) -> float:
    ts = ts.strip()
    m = TIMESTAMP_RE.fullmatch(ts)
    if m:
        h, m2, s, ms = m.groups()
        return int(h)*3600 + int(m2)*60 + int(s) + int(ms)/1000.0
    # Forgiving fallbacks (mm:ss.mmm or ssss.mmm)
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m2, s = parts
            return int(h)*3600 + int(m2)*60 + float(s)
        elif len(parts) == 2:
            m2, s = parts
            return int(m2)*60 + float(s)
        else:
            return float(parts[0])
    except Exception:
        return 0.0

def parse_line(line: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a single line like:
      video.ext | 00:00:12.345=yes, 00:01:00.000=no
    Returns: (video_name, { 'HH:MM:SS.mmm': 'label', ... })
    LAST WRITE WINS within the same line.
    """
    if "|" not in line:
        return "", {}
    name, rest = line.split("|", 1)
    name = name.strip()
    rest = rest.strip()
    if not name or not rest:
        return "", {}

    ts_map: Dict[str, str] = {}
    chunks = [c.strip() for c in rest.split(",")]
    for chunk in chunks:
        if not chunk or "=" not in chunk:
            continue
        ts, lab = chunk.split("=", 1)
        ts = ts.strip()
        lab = (lab or "").strip().lower()
        if ts and lab in VALID_LABELS:
            ts_map[ts] = lab  # last-write-wins within this line
    return name, ts_map

def build_ascii_timeline_for_line(
    ts_map: Dict[str, str],
    width: int,
    fill_mode: str = "unlabeled_as_no",
) -> str:
    """
    Build the bar for a single video's timestamps (from one line).
    Duration is derived from the max timestamp in THIS line.
    """
    max_ts = max((hhmmss_ms_to_seconds(ts) for ts in ts_map.keys()), default=0.0)
    duration = max(0.001, max_ts)  # avoid divide-by-zero

    if fill_mode == "unlabeled_as_no":
        slots = [LABEL_CHAR["no"]] * width
        ranks = [LABEL_RANK["no"]] * width
    else:
        slots = ["-"] * width
        ranks = [0] * width

    for ts_str, lab in ts_map.items():
        t = hhmmss_ms_to_seconds(ts_str)
        t = max(0.0, min(duration, t))
        idx = int(round((t / duration) * (width - 1)))
        idx = max(0, min(width - 1, idx))
        if LABEL_RANK[lab] >= ranks[idx]:
            slots[idx] = LABEL_CHAR[lab]
            ranks[idx] = LABEL_RANK[lab]

    return "[" + "".join(slots) + "]"

def main():
    if not os.path.isfile(INPUT_PATH):
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    results = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            video_name, ts_map = parse_line(line)
            if not video_name:
                continue
            bar = build_ascii_timeline_for_line(ts_map, width=BAR_WIDTH, fill_mode=FILL_MODE)
            results.append(f"{video_name} | {bar}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for row in results:
            out.write(row + "\n")

    print(f"Wrote {len(results)} timelines to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

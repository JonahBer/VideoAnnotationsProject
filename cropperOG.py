#!/usr/bin/env python3
"""
Create cropped highlight videos from annotated timestamps.

INPUT:
- A plain-text data file (DATA_FILE) where each line is:
    <video_filename.mp4> | <HH:MM:SS.mmm>=<label>, <HH:MM:SS.mmm>=<label>, ...
  Example labels: yes, no, perfect (case-insensitive).

ASSUMPTIONS:
- The .mp4 files sit in the same directory as this script (or VIDEO_ROOT).
- The output directory "croppedVideosOG" already exists (or will be created).

CONFIG:
- Edit the CONFIG section below to tune behavior (min perfects, gaps, pre/post roll, etc.)
- Use SOURCE_OFFSET + SOURCE_LIMIT to process a range of source videos (after sorting by filename).
  e.g., SOURCE_OFFSET=3, SOURCE_LIMIT=4  -> process 4 videos starting at index 3 (0-based).
  Set SOURCE_LIMIT = -1 to process all from SOURCE_OFFSET to the end.

DEPENDENCIES:
- FFmpeg and FFprobe must be installed and in PATH (or set FFMPEG_BIN/FFPROBE_BIN).
"""

import re
import csv
import shutil
import math
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

# ----------------------------
# CONFIG (edit these as needed)
# ----------------------------
DATA_FILE = "Charlote_Videos/_frame_annotations2.txt"  # Name of the annotations file
VIDEO_ROOT = "."                      # Directory containing the videos and DATA_FILE
OUTPUT_DIR = "croppedVideosOG"  # Output directory for new videos

# Segment detection
MIN_PERFECTS_PER_SEGMENT = 3          # Minimum number of 'perfect' markers required for a segment
MAX_GAP_BETWEEN_PERFECTS = 10.0       # Seconds between consecutive 'perfect' markers to stay in same cluster
PRE_ROLL = 2.0                        # Seconds before the first perfect in a segment
POST_ROLL = 2.0                       # Seconds after the last perfect in a segment
MERGE_GAP = 3.0                       # Merge segments whose gap is <= this (seconds)
MAX_SEGMENTS_PER_VIDEO = 5            # Limit per source video (set None to disable)
SELECT_TOP_BY = "most_perfects"       # "most_perfects" or "duration"

# Encoding
REENCODE = True                       # True = accurate cuts (re-encode). False = stream copy (keyframe-rounded)
VIDEO_CODEC = "libx264"               # Used if REENCODE=True
CRF = "18"                            # x264 quality (lower=better)
PRESET = "veryfast"                   # x264 preset

FFMPEG_BIN  = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"

DRY_RUN = False                       # True = don't write files, just print actions
WRITE_SUMMARY_CSV = True              # Write a summary CSV of created segments
SUMMARY_CSV_NAME = "crop_summary.csv"

# Range selection of source videos (after sorting by filename, case-insensitive)
SOURCE_OFFSET = 0                     # 0-based start index into the sorted list
SOURCE_LIMIT  = -1                    # -1 = process all from OFFSET; N > 0 = process N videos

# --------------------------------
# Utilities
# --------------------------------
TIME_RE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d{1,3})?)\s*$", re.IGNORECASE)

def parse_timecode(tc: str) -> float:
    m = TIME_RE.match(tc)
    if not m:
        raise ValueError(f"Bad timecode: {tc!r}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss

def secs_to_tc(secs: float) -> str:
    if secs < 0:
        secs = 0.0
    ms = int(round((secs - math.floor(secs)) * 1000))
    s = int(secs) % 60
    m = (int(secs) // 60) % 60
    h = int(secs) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def safe_tag(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())

def clean_token(tok: str) -> str:
    return tok.strip().strip(",").strip(";").strip()

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def has_binary(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None

def get_video_duration(path: Path) -> float:
    """
    Use ffprobe to get video duration in seconds. Returns 0 if it fails or ffprobe missing.
    """
    if not has_binary(FFPROBE_BIN):
        return 0.0
    proc = run([
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# --------------------------------
# Parsing
# --------------------------------
def parse_data_file(fpath: Path) -> Dict[str, List[Tuple[float, str]]]:
    """
    Returns: { 'video.mp4': [(time_sec, label), ... sorted by time] }
    """
    result: Dict[str, List[Tuple[float, str]]] = {}
    if not fpath.exists():
        raise FileNotFoundError(f"Data file not found: {fpath}")

    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                continue
            fname, rest = line.split("|", 1)
            fname = clean_token(fname)
            rest = rest.strip()

            pairs = [clean_token(p) for p in rest.split(",")]
            entries: List[Tuple[float, str]] = []
            for p in pairs:
                if not p or "=" not in p:
                    continue
                left, right = p.rsplit("=", 1)
                t_str = clean_token(left)
                label = safe_tag(clean_token(right))
                if label not in {"yes", "no", "perfect"}:
                    continue
                try:
                    t = parse_timecode(t_str)
                except ValueError:
                    continue
                entries.append((t, label))

            if entries:
                entries.sort(key=lambda x: x[0])
                result[fname] = entries

    return result

# --------------------------------
# Segment builder
# --------------------------------
def build_segments_for_video(entries: List[Tuple[float, str]],
                             min_perfects: int,
                             max_gap_between_perfects: float,
                             pre_roll: float,
                             post_roll: float,
                             merge_gap: float) -> List[Tuple[float, float, int]]:
    """
    Build segments using clusters of 'perfect' markers.
    Returns list of (start_sec, end_sec, perfect_count).
    """
    perfect_times = [t for (t, lbl) in entries if lbl == "perfect"]
    if not perfect_times:
        return []

    # Form clusters of perfects
    clusters: List[List[float]] = []
    cur: List[float] = [perfect_times[0]]
    for prev, now in zip(perfect_times, perfect_times[1:]):
        if (now - prev) <= max_gap_between_perfects:
            cur.append(now)
        else:
            clusters.append(cur)
            cur = [now]
    clusters.append(cur)

    # Convert clusters into segments
    raw_segments: List[Tuple[float, float, int]] = []
    for c in clusters:
        if len(c) >= min_perfects:
            start = c[0] - pre_roll
            end = c[-1] + post_roll
            raw_segments.append((start, end, len(c)))

    if not raw_segments:
        return []

    # Merge segments close in time
    raw_segments.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float, int]] = []
    cur_s, cur_e, cur_cnt = raw_segments[0]
    for s, e, cnt in raw_segments[1:]:
        if s - cur_e <= merge_gap:
            cur_e = max(cur_e, e)
            cur_cnt += cnt
        else:
            merged.append((cur_s, cur_e, cur_cnt))
            cur_s, cur_e, cur_cnt = s, e, cnt
    merged.append((cur_s, cur_e, cur_cnt))
    return merged

def pick_top_segments(segments: List[Tuple[float, float, int]],
                      limit: int,
                      policy: str) -> List[Tuple[float, float, int]]:
    if not segments:
        return []
    if limit is None or limit <= 0 or limit >= len(segments):
        return segments
    if policy == "duration":
        ranked = sorted(segments, key=lambda x: (x[1]-x[0]), reverse=True)
    else:  # "most_perfects"
        ranked = sorted(segments, key=lambda x: (x[2], x[1]-x[0]), reverse=True)
    return sorted(ranked[:limit], key=lambda x: x[0])

# --------------------------------
# Cropping
# --------------------------------
def ffmpeg_crop(input_path: Path, start: float, end: float, out_path: Path) -> Tuple[bool, str]:
    """
    Cut [start, end] inclusive; REENCODE=True for accurate cuts.
    """
    duration = max(0.01, end - start)
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(input_path),
           "-t", f"{duration:.3f}"]
    if REENCODE:
        cmd += ["-c:v", VIDEO_CODEC, "-crf", CRF, "-preset", PRESET, "-c:a", "aac", "-movflags", "+faststart"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-y", str(out_path)]

    if DRY_RUN:
        return True, "DRY_RUN"

    if not has_binary(FFMPEG_BIN):
        return False, f"ffmpeg not found: {FFMPEG_BIN}"

    proc = run(cmd)
    ok = (proc.returncode == 0) and out_path.exists() and out_path.stat().st_size > 0
    msg = proc.stderr.strip() if proc.stderr else "OK"
    return ok, msg

# --------------------------------
# Main
# --------------------------------
def main():
    root = Path(VIDEO_ROOT).resolve()
    data_path = (root / DATA_FILE).resolve()
    out_dir = (root / OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = parse_data_file(data_path)

    # Sort and slice by requested range
    video_items = sorted(annotations.items(), key=lambda kv: kv[0].lower())
    start_idx = max(0, int(SOURCE_OFFSET))
    if SOURCE_LIMIT == -1:
        video_items = video_items[start_idx:]
    else:
        video_items = video_items[start_idx:start_idx + max(0, int(SOURCE_LIMIT))]

    csv_fh = None
    writer = None
    if WRITE_SUMMARY_CSV:
        csv_path = out_dir / SUMMARY_CSV_NAME
        csv_fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_fh)
        writer.writerow(["source_video", "segment_index", "start_sec", "end_sec",
                         "duration_sec", "perfect_count", "output_file"])

    for fname, entries in video_items:
        src_path = root / fname
        if not src_path.exists():
            print(f"[WARN] Missing source video: {src_path}")
            continue

        dur = get_video_duration(src_path)
        segments = build_segments_for_video(
            entries,
            MIN_PERFECTS_PER_SEGMENT,
            MAX_GAP_BETWEEN_PERFECTS,
            PRE_ROLL,
            POST_ROLL,
            MERGE_GAP
        )

        if not segments:
            print(f"[INFO] No qualifying segments in {fname}")
            continue

        segments = pick_top_segments(segments, MAX_SEGMENTS_PER_VIDEO, SELECT_TOP_BY)

        base = src_path.stem
        for i, (s, e, pcnt) in enumerate(segments, start=1):
            if dur > 0:
                s_clamped = clamp(s, 0.0, max(0.0, dur - 0.01))
                e_clamped = clamp(e, s_clamped + 0.01, dur)
            else:
                s_clamped, e_clamped = max(0.0, s), max(0.01, e)

            start_tag = secs_to_tc(s_clamped).replace(":", "-").replace(".", "_")
            end_tag = secs_to_tc(e_clamped).replace(":", "-").replace(".", "_")
            out_name = f"{base}_seg{i}_{start_tag}-{end_tag}.mp4"
            out_path = out_dir / out_name

            ok, msg = ffmpeg_crop(src_path, s_clamped, e_clamped, out_path)
            duration = max(0.0, e_clamped - s_clamped)
            status = "CREATED" if ok else f"FAILED ({msg})"
            print(f"[{status}] {fname} -> {out_name}  [{s_clamped:.3f}–{e_clamped:.3f}s, {duration:.2f}s, perfects={pcnt}]")

            if writer:
                writer.writerow([fname, i, f"{s_clamped:.3f}", f"{e_clamped:.3f}",
                                 f"{duration:.3f}", pcnt, out_name])

    if csv_fh:
        csv_fh.close()
        print(f"[INFO] Summary written to {out_dir / SUMMARY_CSV_NAME}")

if __name__ == "__main__":
    main()

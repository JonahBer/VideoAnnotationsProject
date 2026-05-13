#!/usr/bin/env python3
"""
Create cropped highlight videos from annotated timestamps.

This version organizes outputs into a run folder:
  croppedVideos/<run-name>/

<run-name> is auto-built from config descriptors so runs are easy to differentiate.
The summary CSV is written into that same run folder.

Adjust the CONFIG section to tune behavior and descriptors.

DEPENDENCIES:
- FFmpeg and FFprobe must be installed (or set FFMPEG_BIN/FFPROBE_BIN to full paths).
"""

import re
import csv
import shutil
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

# ----------------------------
# CONFIG (edit these as needed)
# ----------------------------
DATA_FILE   = "Charlote_Videos/_frame_annotations.txt"  # Name of the annotations file
VIDEO_ROOT  = "."                        # Directory containing the videos and DATA_FILE
OUTPUT_DIR  = "croppedVideos"            # Parent output directory

# Segment detection
MIN_PERFECTS_PER_SEGMENT   = 3           # Minimum 'perfect' markers in a cluster
MAX_GAP_BETWEEN_PERFECTS   = 10.0        # Seconds: max gap allowed inside a cluster
PRE_ROLL                    = 2.0         # Seconds before first perfect
POST_ROLL                   = 2.0         # Seconds after last perfect
MERGE_GAP                   = 3.0         # Merge segments if gap <= MERGE_GAP
MAX_SEGMENTS_PER_VIDEO      = 5           # Limit per source video (None to disable)
SELECT_TOP_BY               = "most_perfects"  # or "duration"

# Encoding
REENCODE    = True                       # True=re-encode (accurate). False=stream copy (no quality loss, keyframe-rounded)
VIDEO_CODEC = "libx264"
CRF         = "18"
PRESET      = "veryfast"

# Binaries (your system paths)
FFMPEG_BIN  = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"

DRY_RUN             = False
WRITE_SUMMARY_CSV   = True
SUMMARY_CSV_NAME    = "crop_summary.csv"

# Range selection (after sorting by filename, case-insensitive)
SOURCE_OFFSET = 0       # start index (0-based)
SOURCE_LIMIT  = -1      # -1 = all from offset; N>0 = process N items

# Run naming options (control the folder name inside croppedVideos)
RUN_PREFIX     = "run"          # a short tag you can set per session
ADD_TIMESTAMP  = True           # include datetime in the run name
TIMESTAMP_FMT  = "%Y%m%d-%H%M"  # used if ADD_TIMESTAMP=True

# Which config fields should be included in the run name (descriptor tokens)
INCLUDE_DESCRIPTORS = [
    ("sel",    SELECT_TOP_BY),
    ("minP",   MIN_PERFECTS_PER_SEGMENT),
    ("gap",    MAX_GAP_BETWEEN_PERFECTS),
    ("pre",    PRE_ROLL),
    ("post",   POST_ROLL),
    ("merge",  MERGE_GAP),
    ("maxSeg", MAX_SEGMENTS_PER_VIDEO if MAX_SEGMENTS_PER_VIDEO is not None else "all"),
    ("renc",   int(bool(REENCODE))),
    ("crf",    CRF if REENCODE else "NA"),
    ("off",    SOURCE_OFFSET),
    ("lim",    SOURCE_LIMIT),
]

# --------------------------------
# Utilities
# --------------------------------
def slugify(text: str) -> str:
    # filesystem-safe-ish
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")

TIME_RE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d{1,3})?)\s*$", re.IGNORECASE)

def parse_timecode(tc: str) -> float:
    m = TIME_RE.match(tc)
    if not m:
        raise ValueError(f"Bad timecode: {tc!r}")
    hh = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss

def secs_to_tc(secs: float) -> str:
    if secs < 0: secs = 0.0
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

def has_binary(path: str) -> bool:
    return (Path(path).exists()) or (shutil.which(path) is not None)

def get_video_duration(path: Path) -> float:
    """
    Use ffprobe to get video duration in seconds. Returns 0 if it fails or ffprobe is missing.
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
            if not line or line.startswith("#"): continue
            if "|" not in line: continue

            fname, rest = line.split("|", 1)
            fname = clean_token(fname)
            rest = rest.strip()

            entries: List[Tuple[float, str]] = []
            for p in [clean_token(p) for p in rest.split(",")]:
                if not p or "=" not in p: continue
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
    Build segments using clusters of 'perfect' markers only.
    Returns list of (start_sec, end_sec, perfect_count).
    """
    perfect_times = [t for (t, lbl) in entries if lbl == "perfect"]
    if not perfect_times:
        return []

    # Cluster by time gap
    clusters: List[List[float]] = []
    cur: List[float] = [perfect_times[0]]
    for prev, now in zip(perfect_times, perfect_times[1:]):
        if (now - prev) <= max_gap_between_perfects:
            cur.append(now)
        else:
            clusters.append(cur)
            cur = [now]
    clusters.append(cur)

    # Clusters -> raw segments
    raw_segments: List[Tuple[float, float, int]] = []
    for c in clusters:
        if len(c) >= min_perfects:
            start = c[0] - pre_roll
            end   = c[-1] + post_roll
            raw_segments.append((start, end, len(c)))

    if not raw_segments:
        return []

    # Merge close segments
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
    else:
        ranked = sorted(segments, key=lambda x: (x[2], x[1]-x[0]), reverse=True)
    return sorted(ranked[:limit], key=lambda x: x[0])

# --------------------------------
# FFmpeg crop
# --------------------------------
def ffmpeg_crop(input_path: Path, start: float, end: float, out_path: Path) -> Tuple[bool, str]:
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
def build_run_folder(parent: Path) -> Path:
    tokens = []
    if RUN_PREFIX:
        tokens.append(slugify(RUN_PREFIX))
    # descriptors
    for k, v in INCLUDE_DESCRIPTORS:
        tokens.append(f"{slugify(k)}={slugify(v)}")
    if ADD_TIMESTAMP:
        tokens.append(datetime.now().strftime(TIMESTAMP_FMT))
    run_name = "__".join(tokens)
    run_dir = parent / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # save a small manifest for reference
    (run_dir / "manifest.txt").write_text(
        f"Run folder: {run_name}\nCreated: {datetime.now().isoformat()}\n", encoding="utf-8"
    )
    return run_dir

def main():
    root = Path(VIDEO_ROOT).resolve()
    data_path = (root / DATA_FILE).resolve()
    parent_out = (root / OUTPUT_DIR).resolve()
    parent_out.mkdir(parents=True, exist_ok=True)

    # Build the run folder name (based on config descriptors)
    run_dir = build_run_folder(parent_out)

    annotations = parse_data_file(data_path)

    # Apply source range slice
    video_items = sorted(annotations.items(), key=lambda kv: kv[0].lower())
    start_idx = max(0, int(SOURCE_OFFSET))
    if SOURCE_LIMIT == -1:
        video_items = video_items[start_idx:]
    else:
        video_items = video_items[start_idx:start_idx + max(0, int(SOURCE_LIMIT))]

    csv_fh = None
    writer = None
    if WRITE_SUMMARY_CSV:
        csv_path = run_dir / SUMMARY_CSV_NAME
        csv_fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_fh)
        writer.writerow(["source_video", "segment_index", "start_sec", "end_sec",
                         "duration_sec", "perfect_count", "output_file_rel"])

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
            end_tag   = secs_to_tc(e_clamped).replace(":", "-").replace(".", "_")
            out_name  = f"{base}_seg{i}_{start_tag}-{end_tag}.mp4"
            out_path  = run_dir / out_name

            ok, msg = ffmpeg_crop(src_path, s_clamped, e_clamped, out_path)
            duration = max(0.0, e_clamped - s_clamped)
            status = "CREATED" if ok else f"FAILED ({msg})"
            print(f"[{status}] {fname} -> {out_path.name}  [{s_clamped:.3f}–{e_clamped:.3f}s, {duration:.2f}s, perfects={pcnt}]")

            if writer:
                writer.writerow([fname, i, f"{s_clamped:.3f}", f"{e_clamped:.3f}",
                                 f"{duration:.3f}", pcnt, out_path.name])

    if csv_fh:
        csv_fh.close()
        print(f"[INFO] Summary written to {run_dir / SUMMARY_CSV_NAME}")
        print(f"[INFO] Run folder: {run_dir}")

if __name__ == "__main__":
    main()

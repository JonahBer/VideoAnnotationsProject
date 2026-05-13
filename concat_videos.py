#!/usr/bin/env python3
"""
concat_videos.py — Combine all videos in a folder into one video (robust, no black gaps).

Why black gaps happen:
- Stream-copy concat often breaks at clip boundaries if clips don't start on keyframes and/or have PTS/DTS quirks.
This version defaults to a robust pipeline:
1) Normalize every clip to a common size + CFR fps + sane timestamps (re-encode)
2) Concat them
3) Re-encode the final output (eliminates residual timestamp/keyframe boundary artifacts)

Usage:
  python concat_videos.py "croppedVideos/run__..."                 (default robust mode)
  python concat_videos.py "croppedVideos/run__..." --fps 30 --size 1920x1080

Notes:
- COPY mode is removed on purpose. If you want it back, add it separately; it is the #1 cause of black stretches.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# ------------ CONFIG (paths for your system) ------------
FFMPEG_BIN  = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"

DEFAULT_INPUT_FOLDER = Path(
    "croppedVideos/run__sel=most_perfects__minP=3__gap=10.0__pre=2.0__post=2.0__merge=3.0__maxSeg=5__renc=1__crf=18__off=0__lim=1__20260219-2016"
)

DEFAULT_EXTS    = {".mp4", ".mov", ".mkv", ".m4v"}

VIDEO_CODEC     = "libx264"
CRF             = "18"
PRESET          = "veryfast"

AUDIO_CODEC     = "aac"
AUDIO_RATE      = "48000"
AUDIO_CHANNELS  = "2"
AUDIO_BITRATE   = "192k"
# --------------------------------------------------------


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def has_binary(path: str) -> bool:
    return shutil.which(path) is not None or Path(path).exists()


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_videos(folder: Path, recursive: bool, exts: set[str]) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    files.sort(key=lambda p: natural_key(p.name))
    return files


def ffprobe_props(path: Path) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Returns (width, height, fps) for the first video stream.
    fps is derived from r_frame_rate; may be None if unknown.
    """
    if not has_binary(FFPROBE_BIN):
        return (None, None, None)

    proc = run([
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    if proc.returncode != 0:
        return (None, None, None)

    lines = [x.strip() for x in proc.stdout.strip().splitlines() if x.strip()]
    if len(lines) < 3:
        return (None, None, None)

    try:
        w = int(lines[0])
        h = int(lines[1])
        num, den = lines[2].split("/")
        den_f = float(den)
        fps = float(num) / den_f if den_f != 0 else None
        return (w, h, fps)
    except Exception:
        return (None, None, None)


def normalize_to_tmp(
    inputs: List[Path],
    size: Tuple[int, int],
    fps: Optional[float],
) -> List[Path]:
    """
    Re-encode each input to:
    - common size (letterboxed/padded)
    - CFR (via fps filter if fps provided)
    - sane timestamps (reset in output)
    - H.264 + AAC
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="concat_norm_"))
    norm_paths: List[Path] = []
    w, h = size

    for i, src in enumerate(inputs, start=1):
        dst = tmpdir / f"norm_{i:04d}.mp4"

        vf_chain = [
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
        ]
        if fps is not None:
            vf_chain.append(f"fps={fps:.3f}")  # enforce CFR in filtergraph (preferred)

        cmd = [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "error",
            # make timestamps sane even if source has weirdness
            "-fflags", "+genpts",
            "-i", str(src),
            "-vf", ",".join(vf_chain),
            # video
            "-c:v", VIDEO_CODEC,
            "-crf", CRF,
            "-preset", PRESET,
            # help ensure keyframes reasonably frequent (helps seeking; not required)
            "-g", "60",
            "-keyint_min", "60",
            "-sc_threshold", "0",
            # audio
            "-c:a", AUDIO_CODEC,
            "-ar", AUDIO_RATE,
            "-ac", AUDIO_CHANNELS,
            "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            "-y", str(dst),
        ]

        proc = run(cmd)
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(f"Failed to normalize: {src}\n{proc.stderr.strip()}")

        norm_paths.append(dst)

    return norm_paths


def write_concat_list(paths: List[Path], list_file: Path) -> None:
    with list_file.open("w", encoding="utf-8") as f:
        for p in paths:
            ap = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{ap}'\n")


def concat_and_reencode(
    norm_paths: List[Path],
    out_path: Path,
    fps: Optional[float],
) -> None:
    """
    Concat normalized clips and re-encode the final output.
    Re-encoding final output eliminates black gaps and timestamp discontinuities.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        list_file = Path(tmpdir) / "inputs.txt"
        write_concat_list(norm_paths, list_file)

        cmd = [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            # Optional: if you want to force fps at the end too, uncomment:
            # "-r", f"{fps:.3f}" if fps else "",
            "-c:v", VIDEO_CODEC,
            "-crf", CRF,
            "-preset", PRESET,
            "-c:a", AUDIO_CODEC,
            "-ar", AUDIO_RATE,
            "-ac", AUDIO_CHANNELS,
            "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            "-y", str(out_path),
        ]

        # remove empty args if fps wasn't used above
        cmd = [x for x in cmd if x != ""]

        proc = run(cmd)
        if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(proc.stderr.strip() or "ffmpeg concat failed")


def main() -> int:
    ap = argparse.ArgumentParser(description="Concatenate all videos in a folder into one video (robust).")
    ap.add_argument("folder", nargs="?", default=str(DEFAULT_INPUT_FOLDER),
                    help=f"Folder containing videos (default: {DEFAULT_INPUT_FOLDER})")
    ap.add_argument("--out", type=str, default="combined.mp4",
                    help="Output filename (default: combined.mp4)")
    ap.add_argument("--exts", type=str, default=",".join(sorted(DEFAULT_EXTS)),
                    help="Comma-separated extensions to include (e.g. .mp4,.mov)")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    ap.add_argument("--size", type=str, default=None,
                    help="Target WxH (e.g. 1920x1080). Default: first file's size")
    ap.add_argument("--fps", type=float, default=None,
                    help="Target FPS (CFR). Default: first file's fps if detected; otherwise unset")
    args = ap.parse_args()

    in_folder = Path(args.folder).resolve()
    if not in_folder.exists() or not in_folder.is_dir():
        raise SystemExit(f"Folder not found: {in_folder}")

    if not has_binary(FFMPEG_BIN):
        raise SystemExit(f"ffmpeg not found: {FFMPEG_BIN}")
    if not has_binary(FFPROBE_BIN):
        raise SystemExit(f"ffprobe not found: {FFPROBE_BIN}")

    exts = {
        (e.lower().strip() if e.strip().startswith(".") else "." + e.lower().strip())
        for e in args.exts.split(",") if e.strip()
    }

    inputs = find_videos(in_folder, args.recursive, exts)
    if not inputs:
        raise SystemExit(f"No input videos found in {in_folder}")

    # Output folder: croppedVideos/<input-folder-name>_cropped/
    cropped_parent = Path("./croppedVideos").resolve()
    run_name = in_folder.name
    out_dir = cropped_parent / f"{run_name}_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_filename = Path(args.out).name if args.out.strip() else "combined.mp4"
    out_path = (out_dir / out_filename).resolve()

    # Determine target size / fps
    if args.size:
        try:
            w, h = map(int, args.size.lower().split("x"))
        except Exception:
            raise SystemExit("Invalid --size. Use WxH, e.g. 1920x1080")
    else:
        w0, h0, _fps0 = ffprobe_props(inputs[0])
        if not w0 or not h0:
            raise SystemExit("Could not detect size from first file; pass --size WxH")
        w, h = int(w0), int(h0)

    fps = args.fps
    if fps is None:
        _w0, _h0, det_fps = ffprobe_props(inputs[0])
        fps = det_fps  # may be None
        # If None, we do not force CFR; normalization still fixes most PTS issues.

    # Normalize and concat
    try:
        norm_paths = normalize_to_tmp(inputs, (w, h), fps)
        concat_and_reencode(norm_paths, out_path, fps)
    except Exception as e:
        raise SystemExit(f"Failed: {e}")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
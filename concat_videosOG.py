#!/usr/bin/env python3
"""
concat_videosOG.py — Combine all videos in a folder into one video.

Default behavior:
  - Uses the same folder the other script writes to: ./croppedVideosOG
  - Writes output as ./croppedVideosOG/combined.mp4
  - Tries fast/lossless COPY mode first (requires identical codecs/size/fps). If that fails,
    re-run with --mode reencode (and optionally --size / --fps).

Usage examples:
  # Default (uses ./croppedVideosOG)
  python concat_videosOG.py

  # Specify a different folder and robust re-encode
  python concat_videosOG.py "C:\path\to\some\folder" --mode reencode --size 1920x1080 --fps 30
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# ------------ CONFIG (adjust if you want hard defaults) ------------
# Your installed paths (per your system):
FFMPEG_BIN  = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\bergs\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"

# The default input folder matches the other program's OUTPUT_DIR
DEFAULT_INPUT_FOLDER = Path("croppedVideosOG")

DEFAULT_EXTS = {".mp4", ".mov", ".mkv", ".m4v"}
VIDEO_CODEC = "libx264"
CRF = "18"
PRESET = "veryfast"
AUDIO_CODEC = "aac"
AUDIO_RATE = "48000"   # Hz
AUDIO_CHANNELS = "2"
AUDIO_BITRATE = "192k"
# ---------------------------------------------------------------

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def has_binary(path: str) -> bool:
    return shutil.which(path) is not None or Path(path).exists()

def natural_key(s: str):
    # natural sort helper (e.g., file2 before file10)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def find_videos(folder: Path, recursive: bool, exts: set) -> List[Path]:
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
    Return (width, height, fps) for the primary video stream, or (None,None,None) on failure.
    """
    if not has_binary(FFPROBE_BIN):
        return (None, None, None)
    proc = run([FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    if proc.returncode != 0:
        return (None, None, None)
    lines = [x.strip() for x in proc.stdout.strip().splitlines() if x.strip()]
    if len(lines) < 3:
        return (None, None, None)
    try:
        w = int(lines[0]); h = int(lines[1])
        num, den = lines[2].split("/")
        fps = float(num) / float(den) if float(den) != 0 else None
        return (w, h, fps)
    except Exception:
        return (None, None, None)

def write_list_file(paths: List[Path], list_path: Path):
    # concat demuxer expects: file 'absolute_path'
    with list_path.open("w", encoding="utf-8") as f:
        for p in paths:
            ap = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{ap}'\n")

def concat_copy(paths: List[Path], out_path: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        list_file = Path(tmpdir) / "inputs.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for p in paths:
                ap = str(p.resolve()).replace("'", r"'\''")
                f.write(f"file '{ap}'\n")
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y", str(out_path)
        ]
        proc = run(cmd)
        if proc.returncode != 0:
            print(proc.stderr.strip())
        return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0

def normalize_to_tmp(paths: List[Path], size: Tuple[int,int], fps: Optional[float]) -> List[Path]:
    """
    Re-encode each input to a normalized MP4 in a temp dir, return list of temp paths.
    """
    import tempfile as _tempfile
    tmpdir = Path(_tempfile.mkdtemp(prefix="concat_norm_"))
    norm_paths: List[Path] = []
    w, h = size
    for i, src in enumerate(paths, start=1):
        dst = tmpdir / f"norm_{i:04d}.mp4"
        vf_chain = [
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1"
        ]
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", ",".join(vf_chain),
            "-c:v", VIDEO_CODEC, "-crf", CRF, "-preset", PRESET,
            "-c:a", AUDIO_CODEC, "-ar", AUDIO_RATE, "-ac", AUDIO_CHANNELS, "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
        ]
        if fps:
            cmd += ["-r", f"{fps:.3f}"]
        cmd += ["-y", str(dst)]
        proc = run(cmd)
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(f"Failed to normalize: {src}\n{proc.stderr}")
        norm_paths.append(dst)
    return norm_paths

def main():
    ap = argparse.ArgumentParser(description="Concatenate all videos in a folder into one video.")
    ap.add_argument("folder", nargs="?", default=str(DEFAULT_INPUT_FOLDER),
                    help=f"Folder containing videos (default: {DEFAULT_INPUT_FOLDER})")
    ap.add_argument("--out", type=str, default=None,
                    help="Output filename (default: <folder>/combined.mp4)")
    ap.add_argument("--mode", choices=["copy", "reencode"], default="copy",
                    help="copy = fast (requires identical params); reencode = robust")
    ap.add_argument("--exts", type=str, default=",".join(sorted(DEFAULT_EXTS)),
                    help="Comma-separated extensions to include (e.g. .mp4,.mov)")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    ap.add_argument("--size", type=str, default=None,
                    help="Target WxH for reencode (e.g. 1920x1080). Default: first file's size")
    ap.add_argument("--fps", type=float, default=None,
                    help="Target FPS for reencode. Default: first file's FPS")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    if not has_binary(FFMPEG_BIN):
        raise SystemExit(f"ffmpeg not found: {FFMPEG_BIN}")
    if not has_binary(FFPROBE_BIN):
        raise SystemExit(f"ffprobe not found: {FFPROBE_BIN}")

    # Decide output path
    out_path = Path(args.out).resolve() if args.out else (folder / "combined.mp4").resolve()

    exts = {e.lower().strip() if e.startswith(".") else "." + e.lower().strip()
            for e in args.exts.split(",") if e.strip()}
    inputs = find_videos(folder, args.recursive, exts)
    if not inputs:
        raise SystemExit(f"No input videos found in {folder}")

    if args.mode == "copy":
        print(f"[INFO] Concatenating {len(inputs)} files in COPY mode…")
        ok = concat_copy(inputs, out_path)
        if not ok:
            print("[WARN] COPY mode failed (inputs likely differ in codec/size/fps).")
            print("Try reencode mode, e.g.: --mode reencode --size 1920x1080 --fps 30")
        else:
            print(f"[DONE] {out_path}")
        return

    # reencode mode
    print(f"[INFO] Concatenating {len(inputs)} files in REENCODE mode…")
    # Decide normalization size/fps
    if args.size:
        try:
            w, h = map(int, args.size.lower().split("x"))
        except Exception:
            raise SystemExit("Invalid --size. Use WxH, e.g. 1920x1080")
    else:
        w0, h0, _fps0 = ffprobe_props(inputs[0])
        if not w0 or not h0:
            raise SystemExit("Could not detect size from first file; please pass --size WxH")
        w, h = int(w0), int(h0)

    fps = args.fps
    if fps is None:
        _w0, _h0, det_fps = ffprobe_props(inputs[0])
        fps = det_fps
        if fps is None:
            print("[WARN] Could not detect FPS; leaving unspecified")

    try:
        norm_paths = normalize_to_tmp(inputs, (w, h), fps)
        with tempfile.TemporaryDirectory() as tmpdir:
            list_file = Path(tmpdir) / "inputs.txt"
            with list_file.open("w", encoding="utf-8") as f:
                for p in norm_paths:
                    ap = str(p.resolve()).replace("'", r"'\''")
                    f.write(f"file '{ap}'\n")
            cmd = [
                FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", "-y", str(out_path)
            ]
            proc = run(cmd)
            if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(proc.stderr)
        print(f"[DONE] {out_path}")
    except Exception as e:
        raise SystemExit(f"Failed to concatenate in reencode mode: {e}")

if __name__ == "__main__":
    main()

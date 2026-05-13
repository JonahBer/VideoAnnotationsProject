from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def combine_rating_files(
    src_file_1: str | os.PathLike,
    src_file_2: str | os.PathLike,
    dst_folder: str | os.PathLike,
    out_name: str = "combined.txt",
    default_prefix: str = "Charlote_Videos",
) -> str:
    """
    Combine two ratings text files into one output file.

    Line format:
      <name> | 00:07:38.158=no, 00:23:35.550=yes

    - File1 typically uses basenames: b.MP4
    - File2 may use prefixed paths: Charlote_Videos/b.MP4 (or other dirs)

    Rules:
    - Keep ALL entries from both inputs.
    - Prefer file2 naming (full path) when basenames match.
    - If an entry exists only in file1, synthesize: <default_prefix>/<basename>
    - If the same timestamp appears with different values, keep both by suffixing:
        00:01:02.003=yes, 00:01:02.003#conflict2=no

    Writes: <dst_folder>/<out_name> (out_name is forced to a filename)
    Returns the output file path as a string.
    """
    src1 = Path(src_file_1)
    src2 = Path(src_file_2)
    out_dir = Path(dst_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    def norm_path(s: str) -> str:
        return s.strip().replace("\\", "/")

    def basename_key(name: str) -> str:
        n = norm_path(name)
        return n.split("/")[-1]

    def parse_line(line: str) -> Tuple[str, List[Tuple[str, str]]]:
        line = line.strip()
        if not line:
            return "", []
        if "|" not in line:
            # filename-only line
            return norm_path(line), []
        left, right = line.split("|", 1)
        name = norm_path(left)
        right = right.strip()

        pairs: List[Tuple[str, str]] = []
        if right:
            for chunk in right.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "=" not in chunk:
                    pairs.append((chunk, ""))
                    continue
                ts, val = chunk.split("=", 1)
                pairs.append((ts.strip(), val.strip()))
        return name, pairs

    def parse_file(path: Path) -> Dict[str, List[Tuple[str, str]]]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        data: Dict[str, List[Tuple[str, str]]] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                name, pairs = parse_line(raw)
                if not name:
                    continue
                data.setdefault(name, []).extend(pairs)
        return data

    data1 = parse_file(src1)
    data2 = parse_file(src2)

    # basename -> preferred full name from file2
    preferred_by_base: Dict[str, str] = {}
    for n2 in data2.keys():
        b = basename_key(n2)
        if b not in preferred_by_base:
            preferred_by_base[b] = norm_path(n2)

    def choose_output_name(base: str) -> str:
        if base in preferred_by_base:
            return preferred_by_base[base]
        return f"{norm_path(default_prefix).rstrip('/')}/{base}"

    # merged[out_video_name] = list of (ts, val) across both files
    merged: Dict[str, List[Tuple[str, str]]] = {}

    def add_pairs(out_video_name: str, pairs: List[Tuple[str, str]]) -> None:
        merged.setdefault(out_video_name, []).extend(pairs)

    # Add file2 first under its own names (already preferred)
    for n2, pairs2 in data2.items():
        add_pairs(norm_path(n2), pairs2)

    # Add file1 routed to preferred name if possible
    for n1, pairs1 in data1.items():
        base = basename_key(n1)
        out_video_name = choose_output_name(base)
        add_pairs(out_video_name, pairs1)

    def normalize_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Remove exact duplicates; preserve conflicts by suffixing the timestamp key.
        """
        seen: Dict[str, str] = {}           # ts -> first val
        conflict_count: Dict[str, int] = {} # ts -> count of conflicts emitted
        out: List[Tuple[str, str]] = []

        for ts, val in pairs:
            if not ts:
                continue
            if ts not in seen:
                seen[ts] = val
                out.append((ts, val))
            else:
                if seen[ts] == val:
                    continue
                conflict_count[ts] = conflict_count.get(ts, 1) + 1
                ts2 = f"{ts}#conflict{conflict_count[ts]}"
                out.append((ts2, val))

        out.sort(key=lambda x: x[0])
        return out

    # build output lines
    out_lines: List[str] = []
    for video_name in sorted(merged.keys()):
        pairs = normalize_pairs(merged[video_name])
        if pairs:
            rhs = ", ".join(
                f"{ts}={val}" if val != "" else f"{ts}"
                for ts, val in pairs
            )
            out_lines.append(f"{video_name} | {rhs}")
        else:
            out_lines.append(f"{video_name} |")

    # Force output filename (prevents accidental nested dirs)
    out_filename = Path(out_name).name or "combined.txt"
    out_path = out_dir / out_filename
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return str(out_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Combine two ratings text files.")
    p.add_argument("file1", help="First ratings txt (basename-style names).")
    p.add_argument("file2", help="Second ratings txt (preferred path-style names).")
    p.add_argument("out_dir", help="Destination folder for combined output.")
    p.add_argument("--out-name", default="combined.txt", help="Output filename (default: combined.txt).")
    p.add_argument("--default-prefix", default="Charlote_Videos", help="Prefix for entries only found in file1.")
    args = p.parse_args()

    out_path = combine_rating_files(
        src_file_1=args.file1,
        src_file_2=args.file2,
        dst_folder=args.out_dir,
        out_name=args.out_name,
        default_prefix=args.default_prefix,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
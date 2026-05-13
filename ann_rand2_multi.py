#!/usr/bin/env python3
# annotate_random_frames_multi_fast.py
#
# Multi-folder version tuned to behave like your faster single-folder script:
# - Preload MAIN frames only (fast, always random/bias across videos)
# - ONLY generate "near perfect" followups on-demand AFTER you click Perfect
# - Workers reuse an open cv2.VideoCapture per thread (no reopen when same video repeats)
# - Cache per-video metadata (fps, frame_count) so we don’t re-query every time
# - Keep the same canonical annotation file format (last-write-wins, atomic rewrite)
#
# Requirements: Python 3.9+, pip install opencv-python pillow
from __future__ import annotations

import os
import random
import threading
import queue
from collections import deque
from typing import Dict, List, Tuple, Optional

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ----------------------- Helpers -----------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# Match your "fast" script's weighting behavior: perfect is heavier.
POS_WEIGHT = {"yes": 1, "perfect": 20}
VALID_LABELS = {"yes", "no", "perfect"}


def seconds_to_hhmmss_ms(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    sec_i = int(sec)
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def hhmmss_ms_to_seconds(ts: str) -> float:
    ts = ts.strip()
    if not ts:
        return 0.0
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def _normalize_label(lab: str) -> str:
    lab = (lab or "").strip().lower()
    return lab if lab in VALID_LABELS else ""


def list_videos_recursive(video_dir: str) -> List[str]:
    vids: List[str] = []
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                vids.append(os.path.join(root, fname))
    return vids


def read_existing_annotations_canonical(path: str) -> Dict[str, Dict[str, str]]:
    data: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or "|" not in line:
                    continue
                name, rest = line.split("|", 1)
                name = name.strip().replace("\\", "/")
                rest = rest.strip()
                if not rest:
                    continue
                chunks = [c.strip() for c in rest.split(",")]
                for chunk in chunks:
                    if not chunk or "=" not in chunk:
                        continue
                    ts, lab = chunk.split("=", 1)
                    ts = ts.strip()
                    lab = _normalize_label(lab)
                    if not lab:
                        continue
                    data.setdefault(name, {})[ts] = lab  # last-write-wins
    except Exception:
        return {}
    return data


def write_canonical_annotations_atomic(path: str, data: Dict[str, Dict[str, str]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for name in sorted(data.keys()):
            ts_map = data[name]
            ts_sorted = sorted(ts_map.keys(), key=hhmmss_ms_to_seconds)
            if not ts_sorted:
                continue
            line = f"{name} | " + ", ".join(f"{ts}={ts_map[ts]}" for ts in ts_sorted)
            f.write(line + "\n")
    os.replace(tmp, path)


def resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    if new_size != (w, h):
        return img.resize(new_size, Image.LANCZOS)
    return img


# ----------------------- Reusable cv2.VideoCapture (per worker) -----------------------

class _ReusableCap:
    def __init__(self) -> None:
        self.path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.path = None

    def ensure_open(self, video_path: str) -> cv2.VideoCapture:
        # Reuse if same file still open
        if self.cap is not None and self.path == video_path and self.cap.isOpened():
            return self.cap

        # Switch
        self.close()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.cap = cap
        self.path = video_path
        return cap

    def read_frame(self, frame_idx: int) -> Optional[Image.Image]:
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# ----------------------- App -----------------------

class AnnotatorApp:
    # Preload mains only (fast)
    PRELOAD_MAIN = 40
    NUM_MAIN_WORKERS = 6

    # Followups generated only after Perfect (on main thread, using on-demand read)
    FOLLOWUPS_PER_PERFECT = 5

    # Favoring knobs (same structure as your fast script)
    MIN_GAP_SEC = 1.0
    FAVOR_SIGMA_SEC = 1.75
    FAVOR_POS_PROB = 0.7
    FAVOR_CORRIDOR_PROB = 0.55
    CORRIDOR_WEIGHT_GAMMA = 0.8
    UNSEEN_VIDEO_BONUS = 10.0
    VIDEO_POS_ALPHA = 5.0

    def __init__(self, root: tk.Tk, video_dirs: List[str], annotation_file: str):
        self.root = root
        self.video_dirs = [os.path.abspath(d) for d in video_dirs]
        self.annotation_file = os.path.abspath(annotation_file)

        # ---------------- UI (black theme) ----------------
        root.title("Random Frame Annotator (Multi-folder) (Yes/No/Perfect)")
        root.geometry("1100x800")
        root.configure(bg="black")

        self.info_var = tk.StringVar(value="—")
        self.info_label = tk.Label(root, textvariable=self.info_var, font=("Segoe UI", 12), bg="black", fg="white")
        self.info_label.pack(side=tk.TOP, anchor="w", padx=10, pady=6)

        self.image_label = tk.Label(root, bg="black", bd=0, relief=tk.FLAT, highlightthickness=0)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = tk.Frame(root, bg="black")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.yes_btn = tk.Button(
            btn_frame, text="Yes (1)", font=("Segoe UI", 12), width=14,
            command=lambda: self.record_and_next("yes"),
            bg="black", fg="white", activebackground="#222222", activeforeground="white",
            highlightthickness=0,
        )
        self.no_btn = tk.Button(
            btn_frame, text="No (2)", font=("Segoe UI", 12), width=14,
            command=lambda: self.record_and_next("no"),
            bg="black", fg="white", activebackground="#222222", activeforeground="white",
            highlightthickness=0,
        )
        self.perf_btn = tk.Button(
            btn_frame, text="Perfect (3)", font=("Segoe UI", 12), width=14,
            command=lambda: self.record_and_next("perfect"),
            bg="black", fg="white", activebackground="#222222", activeforeground="white",
            highlightthickness=0,
        )
        self.yes_btn.pack(side=tk.LEFT, padx=5)
        self.no_btn.pack(side=tk.LEFT, padx=5)
        self.perf_btn.pack(side=tk.LEFT, padx=5)

        root.bind("<Key-1>", lambda e: self.record_and_next("yes"))
        root.bind("<Key-2>", lambda e: self.record_and_next("no"))
        root.bind("<Key-3>", lambda e: self.record_and_next("perfect"))
        root.bind("<y>", lambda e: self.record_and_next("yes"))
        root.bind("<n>", lambda e: self.record_and_next("no"))

        # ---------------- Build stable video keys (folder_id/relative_path) ----------------
        self.videos: List[str] = []
        self.video_key_by_path: Dict[str, str] = {}
        self.video_path_by_key: Dict[str, str] = {}

        used_folder_ids: Dict[str, int] = {}
        for d in self.video_dirs:
            if not os.path.isdir(d):
                continue

            base = os.path.basename(os.path.normpath(d)) or "root"
            if base in used_folder_ids:
                used_folder_ids[base] += 1
                folder_id = f"{base}__{used_folder_ids[base]}"
            else:
                used_folder_ids[base] = 1
                folder_id = base

            vids = list_videos_recursive(d)
            for p in vids:
                rel = os.path.relpath(p, d).replace("\\", "/")
                key = f"{folder_id}/{rel}"
                if key in self.video_path_by_key and self.video_path_by_key[key] != p:
                    i = 2
                    new_key = f"{key}__{i}"
                    while new_key in self.video_path_by_key:
                        i += 1
                        new_key = f"{key}__{i}"
                    key = new_key

                self.videos.append(p)
                self.video_key_by_path[p] = key
                self.video_path_by_key[key] = p

        if not self.videos:
            messagebox.showerror("No videos", "No video files found under:\n" + "\n".join(self.video_dirs))
            root.destroy()
            return

        # ---------------- Annotations + caches ----------------
        self.ann_map: Dict[str, Dict[str, str]] = read_existing_annotations_canonical(self.annotation_file)
        try:
            write_canonical_annotations_atomic(self.annotation_file, self.ann_map)
        except Exception:
            pass

        self.annotations_sec: Dict[str, List[Tuple[float, str]]] = {}
        self._rebuild_secondary_caches()

        self.ann_lock = threading.Lock()

        # ---------------- Meta cache (fps, frame_count) ----------------
        self.meta_lock = threading.Lock()
        self.meta_cache: Dict[str, Tuple[int, float]] = {}

        # ---------------- Main preloading ----------------
        self.main_queue: "queue.Queue[dict]" = queue.Queue(maxsize=self.PRELOAD_MAIN)
        self.stop_event = threading.Event()
        self.workers: List[threading.Thread] = []

        # Current shown
        self.current_video_path: str = ""
        self.current_video_key: str = ""
        self.current_timestamp_str: str = ""
        self.current_imgtk = None
        self.current_tag: str = "main"  # "main" or "followup"

        # Followups shown only after perfect
        self.followup_deque = deque()  # holds followup payloads {"video_path","video_key","ts_str","pil_img","tag"}

        # Dedicated UI-thread cap to generate followups without reopening when repeating same video
        self.ui_cap = _ReusableCap()

        # Start workers and show
        self._start_workers()
        self._show_next_frame()

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Caches ----------------

    def _rebuild_secondary_caches(self) -> None:
        self.annotations_sec.clear()
        for vkey, ts_map in self.ann_map.items():
            pairs = [(hhmmss_ms_to_seconds(ts), lab) for ts, lab in ts_map.items()]
            pairs.sort(key=lambda x: x[0])
            self.annotations_sec[vkey] = pairs

    def _get_event_lists(self, video_key: str):
        events = list(self.annotations_sec.get(video_key, []))
        pos = [(s, POS_WEIGHT[l]) for s, l in events if l in POS_WEIGHT]
        all_secs = [s for s, _ in events]
        return pos, all_secs, events

    def _build_pos_corridors(self, video_key: str) -> List[Tuple[float, float, float]]:
        _, _, events = self._get_event_lists(video_key)
        corridors: List[Tuple[float, float, float]] = []
        pos_indices = [i for i, (_, lab) in enumerate(events) if lab in POS_WEIGHT]
        for a, b in zip(pos_indices, pos_indices[1:]):
            has_no = any(lab == "no" for _, lab in events[a + 1 : b])
            if has_no:
                continue
            start = events[a][0]
            end = events[b][0]
            if end <= start:
                continue
            w_a = POS_WEIGHT[events[a][1]]
            w_b = POS_WEIGHT[events[b][1]]
            corridors.append((start, end, (w_a + w_b) / 2.0))
        return corridors

    # ---------------- Meta ----------------

    def _get_meta_cached(self, cap_reuse: _ReusableCap, video_path: str) -> Optional[Tuple[int, float]]:
        with self.meta_lock:
            got = self.meta_cache.get(video_path)
        if got is not None:
            return got

        try:
            cap = cap_reuse.ensure_open(video_path)
        except Exception:
            return None

        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fc <= 0 or fps <= 0:
            return None

        with self.meta_lock:
            self.meta_cache[video_path] = (fc, fps)
        return (fc, fps)

    # ---------------- Biasing / sampling ----------------

    def _uniform_sec_away_from_events(self, avoid_secs: List[float], duration: float, rnd: random.Random, tries: int = 50) -> float:
        lo = max(0.02 * duration, 0.0)
        hi = max(0.0, 0.98 * duration)
        for _ in range(tries):
            t = lo + rnd.random() * (hi - lo)
            if not avoid_secs:
                return t
            if min(abs(t - s) for s in avoid_secs) >= self.MIN_GAP_SEC:
                return t
        return lo + rnd.random() * (hi - lo)

    def _pick_biased_second(self, video_key: str, frame_count: int, fps: float, rnd: random.Random) -> float:
        duration = frame_count / max(fps, 1e-9)
        pos_list, all_secs, _ = self._get_event_lists(video_key)

        corridors = self._build_pos_corridors(video_key)
        if corridors and rnd.random() < self.FAVOR_CORRIDOR_PROB:
            lengths = [(b - a) for (a, b, _) in corridors]
            weights = [max(L, 1e-3) ** self.CORRIDOR_WEIGHT_GAMMA * max(ep, 0.5)
                       for L, (_, _, ep) in zip(lengths, corridors)]
            (a, b, _) = rnd.choices(corridors, weights=weights, k=1)[0]
            lo_c = a + self.MIN_GAP_SEC
            hi_c = b - self.MIN_GAP_SEC
            if hi_c > lo_c:
                for _ in range(24):
                    t = lo_c + rnd.random() * (hi_c - lo_c)
                    nearest = min(abs(t - s) for s in all_secs) if all_secs else float("inf")
                    if nearest >= self.MIN_GAP_SEC:
                        return t

        if pos_list and rnd.random() < self.FAVOR_POS_PROB:
            secs = [s for s, _ in pos_list]
            wts = [max(w, 1e-3) for _, w in pos_list]
            lo = max(0.02 * duration, 0.0)
            hi = max(0.0, 0.98 * duration)
            for _ in range(48):
                center = rnd.choices(secs, weights=wts, k=1)[0]
                t = center + rnd.gauss(0.0, self.FAVOR_SIGMA_SEC)
                t = min(max(t, lo), hi)
                nearest = min(abs(t - s) for s in all_secs) if all_secs else float("inf")
                if nearest >= self.MIN_GAP_SEC:
                    return t

        return self._uniform_sec_away_from_events(all_secs, duration, rnd)

    # ---------------- Frame reading (reuse cap) ----------------

    def _read_frame_at_second_reuse(self, cap_reuse: _ReusableCap, video_path: str, target_sec: float, jitter_attempts: int = 2):
        cap = cap_reuse.ensure_open(video_path)

        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fc <= 0 or fps <= 0:
            raise RuntimeError("Invalid meta")

        duration = fc / fps
        lo = max(0.02 * duration, 0.0)
        hi = max(0.0, 0.98 * duration)
        t = min(max(target_sec, lo), hi)

        for attempt in range(jitter_attempts + 1):
            idx = int(round(t * fps))
            idx = max(0, min(fc - 1, idx))

            img = cap_reuse.read_frame(idx)
            if img is not None:
                ts_seconds = idx / fps
                ts_str = seconds_to_hhmmss_ms(ts_seconds)
                return img, ts_str, ts_seconds

            t = min(max(t + (0.5 * (1 if attempt % 2 == 0 else -1)), lo), hi)

        raise RuntimeError("Failed to read")

    def _get_random_frame_reuse(self, cap_reuse: _ReusableCap, video_path: str, max_attempts: int = 3):
        cap = cap_reuse.ensure_open(video_path)

        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fc <= 0 or fps <= 0:
            raise RuntimeError("Invalid meta")

        lo_idx = max(0, int(0.02 * fc))
        hi_idx = max(0, int(0.98 * fc))

        for _ in range(max_attempts):
            idx = random.randint(lo_idx, hi_idx)
            img = cap_reuse.read_frame(idx)
            if img is not None:
                ts_seconds = idx / fps
                ts_str = seconds_to_hhmmss_ms(ts_seconds)
                return img, ts_str, ts_seconds

        raise RuntimeError("Failed random read")

    # ---------------- Video choice (weights) ----------------

    def _choose_video_with_weights(self, rnd: random.Random) -> str:
        weights: List[float] = []
        with self.ann_lock:
            for p in self.videos:
                key = self.video_key_by_path[p]
                ts_map = self.ann_map.get(key, {})
                if not ts_map:
                    w = 1.0 + self.UNSEEN_VIDEO_BONUS
                else:
                    pos_units = sum(POS_WEIGHT.get(lab, 0) for lab in ts_map.values())
                    w = 1.0 + self.VIDEO_POS_ALPHA * pos_units
                weights.append(max(w, 0.001))
        return rnd.choices(self.videos, weights=weights, k=1)[0]

    # ---------------- Workers (preload mains only) ----------------

    def _main_worker(self):
        rnd = random.Random()
        cap_reuse = _ReusableCap()
        try:
            while not self.stop_event.is_set():
                if self.main_queue.full():
                    self.stop_event.wait(0.05)
                    continue

                try:
                    with self.ann_lock:
                        vid_path = self._choose_video_with_weights(rnd)
                        vid_key = self.video_key_by_path[vid_path]
                        all_secs_snapshot = [s for s, _ in self.annotations_sec.get(vid_key, [])]
                except Exception:
                    continue

                meta = self._get_meta_cached(cap_reuse, vid_path)
                if not meta:
                    continue
                frame_count, fps = meta

                # try biased picks a few times; fallback to random
                pil_img = None
                ts_str = ""
                ts_sec = 0.0
                for _ in range(8):
                    try:
                        target = self._pick_biased_second(vid_key, frame_count, fps, rnd)
                        img, ts_s, ts_seconds = self._read_frame_at_second_reuse(cap_reuse, vid_path, target, jitter_attempts=1)
                    except Exception:
                        continue
                    nearest = min(abs(ts_seconds - s) for s in all_secs_snapshot) if all_secs_snapshot else float("inf")
                    if nearest >= self.MIN_GAP_SEC:
                        pil_img, ts_str, ts_sec = img, ts_s, ts_seconds
                        break

                if pil_img is None:
                    try:
                        pil_img, ts_str, ts_sec = self._get_random_frame_reuse(cap_reuse, vid_path, max_attempts=2)
                    except Exception:
                        continue

                payload = {
                    "video_path": vid_path,
                    "video_key": vid_key,
                    "ts_str": ts_str,
                    "ts_sec": ts_sec,
                    "pil_img": pil_img,
                    "tag": "main",
                }

                try:
                    self.main_queue.put(payload, timeout=0.1)
                except queue.Full:
                    pass
        finally:
            cap_reuse.close()

    def _start_workers(self):
        for i in range(self.NUM_MAIN_WORKERS):
            t = threading.Thread(target=self._main_worker, name=f"main-worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

    # ---------------- Followups (on-demand, only after Perfect) ----------------

    def _generate_followups_on_demand(self, vid_path: str, vid_key: str, center_sec: float) -> List[dict]:
        """
        Create up to FOLLOWUPS_PER_PERFECT frames near center_sec, respecting MIN_GAP vs existing annotations.
        Uses the UI-thread reusable cap so repeated followups on the same video do not reopen the file.
        """
        meta = self._get_meta_cached(self.ui_cap, vid_path)
        if not meta:
            return []
        frame_count, fps = meta
        duration = frame_count / max(fps, 1e-9)
        lo = max(0.02 * duration, 0.0)
        hi = max(0.0, 0.98 * duration)

        # Snapshot existing annotations seconds for this video
        with self.ann_lock:
            all_secs = [s for s, _ in self.annotations_sec.get(vid_key, [])]

        rnd = random.Random()

        base_radii = [
            self.MIN_GAP_SEC * 1.10,
            max(0.75, self.MIN_GAP_SEC * 1.50),
            max(1.25, self.MIN_GAP_SEC * 2.00),
            max(2.00, self.MIN_GAP_SEC * 3.00),
            max(3.00, self.MIN_GAP_SEC * 4.50),
            max(4.50, self.MIN_GAP_SEC * 6.50),
        ]

        out: List[dict] = []
        picked_ts: set[str] = set()
        picked_secs: List[float] = []

        attempts = 0
        while len(out) < self.FOLLOWUPS_PER_PERFECT and attempts < 160:
            attempts += 1
            r = rnd.choice(base_radii)
            sign = 1.0 if rnd.random() < 0.5 else -1.0
            jitter = rnd.uniform(-0.35, 0.35)
            t = center_sec + sign * r + jitter
            t = min(max(t, lo), hi)

            try:
                img, ts_str, ts_sec = self._read_frame_at_second_reuse(self.ui_cap, vid_path, t, jitter_attempts=1)
            except Exception:
                continue

            if ts_str in picked_ts:
                continue

            # gap vs existing annotations
            nearest_existing = min(abs(ts_sec - s) for s in all_secs) if all_secs else float("inf")
            if nearest_existing < self.MIN_GAP_SEC:
                continue

            # gap vs other followups (avoid near-duplicates)
            if picked_secs:
                nearest_prev = min(abs(ts_sec - s) for s in picked_secs)
                if nearest_prev < max(0.50, self.MIN_GAP_SEC * 0.75):
                    continue

            picked_ts.add(ts_str)
            picked_secs.append(ts_sec)
            out.append({"video_path": vid_path, "video_key": vid_key, "ts_str": ts_str, "pil_img": img, "tag": "followup"})

        return out

    # ---------------- Display ----------------

    def _display_payload(self, item: dict, extra_note: str = ""):
        self.current_video_path = item["video_path"]
        self.current_video_key = item["video_key"]
        self.current_timestamp_str = item["ts_str"]
        self.current_tag = item.get("tag", "main")

        self.root.update_idletasks()
        max_w = max(200, self.image_label.winfo_width() or 1000)
        max_h = max(200, self.image_label.winfo_height() or 700)

        disp_img = resize_to_fit(item["pil_img"], max_w, max_h)
        self.current_imgtk = ImageTk.PhotoImage(disp_img)
        self.image_label.configure(image=self.current_imgtk)

        note = f"   {extra_note}" if extra_note else ""
        self.info_var.set(f"Video: {self.current_video_key}   |   Timestamp: {self.current_timestamp_str}{note}")

    def _show_next_frame(self):
        # If followups are queued (only after perfect), drain them first
        if self.followup_deque:
            item = self.followup_deque.popleft()
            self._display_payload(item, extra_note="[near perfect]")
            return

        # Otherwise show next main
        try:
            item = self.main_queue.get_nowait()
        except queue.Empty:
            self.info_var.set("Loading next frame…")
            self.root.after(50, self._show_next_frame)
            return

        self._display_payload(item)

    # ---------------- Record ----------------

    def record_and_next(self, label: str):
        if not self.current_video_key or not self.current_timestamp_str:
            self._show_next_frame()
            return

        lab = _normalize_label(label)
        if not lab:
            self._show_next_frame()
            return

        vkey = self.current_video_key
        ts_str = self.current_timestamp_str

        # Save annotation
        with self.ann_lock:
            prev = self.ann_map.setdefault(vkey, {}).get(ts_str)
            if prev != lab:
                self.ann_map[vkey][ts_str] = lab
                self._rebuild_secondary_caches()
                try:
                    write_canonical_annotations_atomic(self.annotation_file, self.ann_map)
                except Exception as e:
                    messagebox.showwarning("Save warning", f"Could not write annotations:\n{e}")

        # Only after PERFECT on a MAIN frame do we show more frames from the same video.
        # Yes/No always returns to global main preloaded queue.
        if lab == "perfect" and self.current_tag == "main":
            # generate 5 near-perfect frames right now (on demand)
            # (fast because we reuse the UI cap and cached meta)
            try:
                # center is current timestamp in seconds; compute from string
                center_sec = hhmmss_ms_to_seconds(self.current_timestamp_str)
                followups = self._generate_followups_on_demand(
                    vid_path=self.current_video_path,
                    vid_key=self.current_video_key,
                    center_sec=center_sec,
                )
            except Exception:
                followups = []

            self.followup_deque.clear()
            for fu in followups:
                self.followup_deque.append(fu)

        self._show_next_frame()

    def on_close(self):
        self.stop_event.set()
        for t in self.workers:
            t.join(timeout=0.2)
        try:
            self.ui_cap.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    VIDEO_DIRS = [
        r"D:\NewFolder(3)\Grace\All Content\Videos",
        r"C:\Users\bergs\videoProject\Charlote_Videos",
        r"C:\Users\bergs\videoProject\Gretas_Videos",
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_file = os.path.join(script_dir, "_frame_annotations.txt")

    ok_dirs = [d for d in VIDEO_DIRS if os.path.isdir(d)]
    if not ok_dirs:
        raise SystemExit("None of the VIDEO_DIRS exist:\n" + "\n".join(VIDEO_DIRS))

    root = tk.Tk()

    AnnotatorApp(root, ok_dirs, annotation_file)
    root.mainloop()


if __name__ == "__main__":
    main()

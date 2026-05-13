# annotate_random_frames.py
# Requirements (Windows): Python 3.9+, pip install opencv-python pillow
# Persistent Tkinter UI for Yes/No/Perfect frame annotation.
# Fixes:
#   • Canonical, de-duplicated annotation store (last-write-wins) with atomic rewrites (no more corrupt/duplicated lines).
#   • "perfect" counts double (not 50x).
#   • Global MIN_GAP from ANY prior label (yes/no/perfect) when sampling new frames.
#   • Corridor/positive sampling respects MIN_GAP and avoids endpoints.
#   • Startup auto-clean of existing file to canonical form.
from __future__ import annotations

import os
import random
import threading
import queue
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from typing import Dict, List, Tuple

# ----------------------- Helpers -----------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
POS_WEIGHT = {"yes": 1, "perfect": 20}  # perfect counts double (not 50)
VALID_LABELS = {"yes", "no", "perfect"}

def list_videos(video_dir: str) -> List[str]:
    """Recursively list all videos under video_dir."""
    vids: List[str] = []
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                vids.append(os.path.join(root, fname))
    return vids

def seconds_to_hhmmss_ms(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def hhmmss_ms_to_seconds(ts: str) -> float:
    ts = ts.strip()
    if not ts:
        return 0.0
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        sec = float(s)
        return int(h) * 3600 + int(m) * 60 + sec
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])

def _normalize_label(lab: str) -> str:
    lab = (lab or "").strip().lower()
    return lab if lab in VALID_LABELS else ""

def read_existing_annotations_canonical(path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse any mix of:
      video.ext | 00:00:12.345=yes
      subdir/video.ext | 00:00:12.345=yes, 00:01:00.000=no
    Returns canonical mapping with last-write-wins:
      { rel_video_path: { "hh:mm:ss.mmm": "label", ... } }
    """
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
                name = name.strip().replace("\\", "/")  # normalize separators
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
                    # last-write-wins
                    data.setdefault(name, {})[ts] = lab
    except Exception:
        data = {}
    return data

def write_canonical_annotations_atomic(path: str, data: Dict[str, Dict[str, str]]) -> None:
    """
    Write canonical file:
      rel/path/filename | ts1=lab, ts2=lab, ...
    Sorted by filename then timestamp ascending. Atomic replace.
    """
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

def get_video_meta(video_path: str) -> Tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return frame_count, fps

def read_frame_at_second(video_path: str, target_sec: float, jitter_attempts: int = 4):
    """
    Try to grab frame at/near target_sec; fall back by small jitters.
    Returns (PIL.Image, timestamp_str, timestamp_seconds).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    duration = frame_count / fps
    lo = max(0.02 * duration, 0.0)
    hi = max(0.0, 0.98 * duration)
    t = min(max(target_sec, lo), hi)

    for attempt in range(jitter_attempts + 1):
        idx = int(round(t * fps))
        idx = max(0, min(frame_count - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ts_seconds = idx / fps
            ts_str = seconds_to_hhmmss_ms(ts_seconds)
            cap.release()
            return pil_img, ts_str, ts_seconds
        # small local jitter: +/- 0.5s then +/-1.0s...
        t = min(max(t + (0.5 * (1 if attempt % 2 == 0 else -1)), lo), hi)

    cap.release()
    raise RuntimeError(f"Failed to read a specific frame from: {video_path}")

def get_random_frame(video_path: str, max_attempts: int = 5):
    """
    Uniform random fallback. Returns (PIL.Image, timestamp_str, timestamp_seconds).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    for _ in range(max_attempts):
        idx = random.randint(max(0, int(0.02 * frame_count)), max(0, int(0.98 * frame_count)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ts_seconds = idx / fps
            ts_str = seconds_to_hhmmss_ms(ts_seconds)
            cap.release()
            return pil_img, ts_str, ts_seconds

    cap.release()
    raise RuntimeError(f"Failed to read a random frame from: {video_path}")

def resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    if new_size != (w, h):
        return img.resize(new_size, Image.LANCZOS)
    return img

# ----------------------- UI App -----------------------

class AnnotatorApp:
    # Queue / threads
    PRELOAD_MAX = 20
    NUM_WORKERS = 4

    # Favoring knobs
    MIN_GAP_SEC = 1.0            # avoid ANY prior label within this many seconds
    FAVOR_SIGMA_SEC = 1.75       # Gaussian spread around a positive label
    FAVOR_POS_PROB = 0.7         # when not in corridor mode, probability to bias near a positive label
    FAVOR_CORRIDOR_PROB = 0.55   # probability to sample inside a corridor (pos..pos with no "no")
    CORRIDOR_WEIGHT_GAMMA = 0.8  # weight corridors by (length**gamma) * endpoint_weight_factor
    UNSEEN_VIDEO_BONUS = 1.0    # additive weight for videos with zero annotations
    VIDEO_POS_ALPHA = 1.0        # additive weight per unit of positive weight on a video (yes=1, perfect=2)

    LOW_RATING_BOOST = 6.0  # bigger => more bias toward low-rated videos
    LOW_RATING_POWER = 1.75  # bigger => bias concentrates more on the lowest-rated
    RATING_SMOOTHING = 3.0  # prevents tiny sample counts from being extreme
    BASE_VIDEO_WEIGHT = 1.0  # baseline for all videos

    def __init__(self, root: tk.Tk, video_dir: str):
        self.root = root
        self.video_dir = os.path.abspath(video_dir)
        self.annotation_file = os.path.join(self.video_dir, "_frame_annotations.txt")



        # ---- Cache videos (RECURSIVE)
        self.videos: List[str] = list_videos(self.video_dir)
        if not self.videos:
            messagebox.showerror("No videos", f"No video files found under:\n{self.video_dir}")
            root.destroy()
            return

        # Use RELATIVE PATHS (forward slashes) as stable keys to avoid collisions across subfolders.
        def _relkey(p: str) -> str:
            rel = os.path.relpath(p, self.video_dir)
            return rel.replace("\\", "/")

        self.video_name_by_path: Dict[str, str] = {p: _relkey(p) for p in self.videos}
        self.video_path_by_name: Dict[str, str] = {vname: p for p, vname in self.video_name_by_path.items()}

        # ---- Canonical annotations: {rel_path: {ts_str: label}}
        self.ann_map: Dict[str, Dict[str, str]] = read_existing_annotations_canonical(self.annotation_file)

        # Migrate old basename-only keys when unambiguous
        self._migrate_basename_keys_if_unambiguous()

        # Cleanup the file on startup (dedupe + sort)
        try:
            write_canonical_annotations_atomic(self.annotation_file, self.ann_map)
        except Exception:
            pass

        # Derived caches
        self.annotations_sec: Dict[str, List[Tuple[float, str]]] = {}
        self.annotation_events: List[Tuple[str, str, str]] = []  # (rel_path, ts_str, label)
        self._rebuild_secondary_caches()

        self.ann_lock = threading.Lock()  # guard annotations structures

        # Current context
        self.current_video_path: str = ""
        self.current_video_name: str = ""
        self.current_timestamp_str: str = ""
        self.current_imgtk = None  # keep ref

        self._pending_followup: Tuple[str, str] | None = None  # (video_name, center_ts_str)

        # Preloader infra
        self.frame_queue: "queue.Queue[dict]" = queue.Queue(maxsize=self.PRELOAD_MAX)
        self.stop_event = threading.Event()
        self.workers: List[threading.Thread] = []

        # Window
        root.title("Random Frame Annotator (Yes/No/Perfect)")
        root.geometry("1100x800")  # resizable

        # Top info
        self.info_var = tk.StringVar(value="—")
        self.info_label = tk.Label(root, textvariable=self.info_var, font=("Segoe UI", 12))
        self.info_label.pack(side=tk.TOP, anchor="w", padx=10, pady=6)

        # Image area
        self.image_label = tk.Label(root, bd=1, relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.yes_btn = tk.Button(btn_frame, text="Yes (1)", font=("Segoe UI", 12), width=14, command=lambda: self.record_and_next("yes"))
        self.no_btn  = tk.Button(btn_frame, text="No (2)",  font=("Segoe UI", 12), width=14, command=lambda: self.record_and_next("no"))
        self.perf_btn= tk.Button(btn_frame, text="Perfect (3)", font=("Segoe UI", 12), width=14, command=lambda: self.record_and_next("perfect"))

        self.yes_btn.pack(side=tk.LEFT, padx=5)
        self.no_btn.pack(side=tk.LEFT, padx=5)
        self.perf_btn.pack(side=tk.LEFT, padx=5)

        # Keyboard shortcuts
        root.bind("<y>", lambda e: self.record_and_next("yes"))
        root.bind("<n>", lambda e: self.record_and_next("no"))
        root.bind("<Key-1>", lambda e: self.record_and_next("yes"))
        root.bind("<Key-2>", lambda e: self.record_and_next("no"))
        root.bind("<Key-3>", lambda e: self.record_and_next("perfect"))

        # Start preloader workers
        self._start_preloader_workers()

        # First frame (non-blocking poll until a preloaded frame is available)
        self._show_next_preloaded_frame()

        # Handle close to flush and stop workers
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _video_rating_0to1(self, vid_name: str) -> float:
        """
        Derive a per-video rating in [0..1] from existing annotations.
        Uses your POS_WEIGHT mapping for yes/perfect.
        """
        ts_map = self.ann_map.get(vid_name, {})
        if not ts_map:
            return 0.5  # neutral if unseen (weighting handled separately)

        pos_units = 0.0
        no_count = 0.0
        for lab in ts_map.values():
            if lab == "no":
                no_count += 1.0
            else:
                pos_units += float(POS_WEIGHT.get(lab, 0))

        denom = pos_units + no_count + float(self.RATING_SMOOTHING)
        if denom <= 0:
            return 0.5
        return max(0.0, min(1.0, pos_units / denom))

    def _video_pick_weight(self, vid_name: str) -> float:
        """
        Convert rating -> pick weight (lower rating => higher weight).
        """
        ts_map = self.ann_map.get(vid_name, {})
        if not ts_map:
            return float(self.BASE_VIDEO_WEIGHT + self.UNSEEN_VIDEO_BONUS)

        rating = self._video_rating_0to1(vid_name)          # 0..1
        low_factor = (1.0 - rating) ** float(self.LOW_RATING_POWER)
        return float(self.BASE_VIDEO_WEIGHT + self.LOW_RATING_BOOST * low_factor)


    # -------- Migration helper (old -> new keys) --------
    def _migrate_basename_keys_if_unambiguous(self):
        """If ann_map uses bare basenames and there's exactly one match in current tree, remap to rel path."""
        # Build basename -> [rel_keys...] map
        basename_index: Dict[str, List[str]] = {}
        for rel_key in self.video_path_by_name.keys():
            base = os.path.basename(rel_key)
            basename_index.setdefault(base, []).append(rel_key)

        # Collect remaps
        remaps: List[Tuple[str, str]] = []
        for key in list(self.ann_map.keys()):
            if key in self.video_path_by_name:
                continue  # already a rel path key we know
            base = os.path.basename(key)
            candidates = basename_index.get(base, [])
            if len(candidates) == 1:
                remaps.append((key, candidates[0]))

        # Apply remaps
        for old, new in remaps:
            existing = self.ann_map.pop(old, {})
            # Merge if needed, last-write-wins already applied at load
            tgt = self.ann_map.setdefault(new, {})
            tgt.update(existing)

    # -------- Annotation caches --------

    def _rebuild_secondary_caches(self):
        self.annotations_sec.clear()
        self.annotation_events.clear()
        for vid_name, ts_map in self.ann_map.items():
            pairs_sec = []
            for ts_str, lab in ts_map.items():
                pairs_sec.append((hhmmss_ms_to_seconds(ts_str), lab))
                self.annotation_events.append((vid_name, ts_str, lab))
            # sort by seconds for stable behavior
            pairs_sec.sort(key=lambda x: x[0])
            self.annotations_sec[vid_name] = pairs_sec

    # -------- Annotation utilities --------

    def _get_event_lists(self, video_name: str):
        """Return (pos_list[(sec,weight)], no_secs, all_secs_sorted, events_sorted_by_sec[(sec,label)]) for a video."""
        pairs = list(self.annotations_sec.get(video_name, []))
        events = sorted(pairs, key=lambda x: x[0])
        pos = [(s, POS_WEIGHT[l]) for s, l in events if l in POS_WEIGHT]
        no_secs = [s for s, l in events if l == "no"]
        all_secs = [s for s, _ in events]
        return pos, no_secs, all_secs, events

    def _build_pos_corridors(self, video_name: str) -> List[Tuple[float, float, float]]:
        """
        Corridors: (start_sec, end_sec, endpoint_weight_factor)
        where both ends are positive (yes/perfect) and there is NO "no" between them.
        endpoint_weight_factor = average of endpoint positive weights.
        """
        _, _, _, events = self._get_event_lists(video_name)
        corridors: List[Tuple[float, float, float]] = []
        pos_indices = [i for i, (_, lab) in enumerate(events) if lab in POS_WEIGHT]
        for a, b in zip(pos_indices, pos_indices[1:]):
            # Ensure no 'no' in (a, b)
            has_no = any(lab == "no" for _, lab in events[a+1:b])
            if not has_no:
                start = events[a][0]
                end   = events[b][0]
                if end > start:
                    w_a = POS_WEIGHT[events[a][1]]
                    w_b = POS_WEIGHT[events[b][1]]
                    corridors.append((start, end, (w_a + w_b) / 2.0))
        return corridors

    # -------- Favoring logic --------

    def _uniform_sec_away_from_events(self, avoid_secs: List[float], duration: float, rnd: random.Random, tries: int = 50) -> float:
        """Sample uniformly in [2%, 98%] duration but reject within MIN_GAP_SEC of ANY prior label."""
        lo = max(0.02 * duration, 0.0)
        hi = max(0.0, 0.98 * duration)
        for _ in range(tries):
            t = lo + rnd.random() * (hi - lo)
            if not avoid_secs:
                return t
            nearest = min(abs(t - s) for s in avoid_secs)
            if nearest >= self.MIN_GAP_SEC:
                return t
        # If we keep failing, return best-effort
        return lo + rnd.random() * (hi - lo)

    def _pick_biased_second(self, video_name: str, frame_count: int, fps: float, rnd: random.Random) -> float:
        """
        Choose a timestamp (sec) given prior labels for a video:
          1) With prob FAVOR_CORRIDOR_PROB, sample inside a corridor [pos .. pos] with no "no" inside,
             uniformly but shrink by MIN_GAP_SEC away from endpoints; reject if too close to ANY event.
          2) Else with prob FAVOR_POS_PROB, sample near a prior positive label (Gaussian),
             selecting centers with weights (yes=1, perfect=2). Reject if too close to ANY event.
          3) Else sample uniformly away from ANY event by MIN_GAP_SEC.
        """
        duration = frame_count / max(fps, 1e-9)
        pos_list, _, all_secs, _ = self._get_event_lists(video_name)

        # 1) Corridor sampling
        corridors = self._build_pos_corridors(video_name)
        if corridors and rnd.random() < self.FAVOR_CORRIDOR_PROB:
            # Weight by length**gamma times endpoint weight factor
            lengths = [(b - a) for (a, b, _) in corridors]
            weights = [max(L, 1e-3) ** self.CORRIDOR_WEIGHT_GAMMA * max(ep, 0.5) for L, (_, _, ep) in zip(lengths, corridors)]
            if hasattr(rnd, "choices"):
                (a, b, ep) = rnd.choices(corridors, weights=weights, k=1)[0]
            else:
                total = sum(weights)
                r = rnd.random() * total
                acc = 0.0
                pick = corridors[-1]
                for c, w in zip(corridors, weights):
                    acc += w
                    if r <= acc:
                        pick = c
                        break
                (a, b, ep) = pick
            lo_c = a + self.MIN_GAP_SEC
            hi_c = b - self.MIN_GAP_SEC
            if hi_c > lo_c:
                for _ in range(24):
                    t = lo_c + rnd.random() * (hi_c - lo_c)
                    nearest = min(abs(t - s) for s in all_secs) if all_secs else float("inf")
                    if nearest >= self.MIN_GAP_SEC:
                        return t
            # If corridor too tight, fall through

        # 2) Around a single positive label
        if pos_list and rnd.random() < self.FAVOR_POS_PROB:
            secs = [s for s, _ in pos_list]
            wts  = [max(w, 1e-3) for _, w in pos_list]  # yes=1, perfect=2
            lo = max(0.02 * duration, 0.0)
            hi = max(0.0, 0.98 * duration)
            for _ in range(48):  # rejection attempts if too close to ANY event
                if hasattr(rnd, "choices"):
                    center = rnd.choices(secs, weights=wts, k=1)[0]
                else:
                    total = sum(wts)
                    r = rnd.random() * total
                    acc = 0.0
                    center = secs[-1]
                    for s, w in zip(secs, wts):
                        acc += w
                        if r <= acc:
                            center = s
                            break
                t = center + rnd.gauss(0.0, self.FAVOR_SIGMA_SEC)
                t = min(max(t, lo), hi)
                nearest = min(abs(t - s) for s in all_secs) if all_secs else float("inf")
                if nearest >= self.MIN_GAP_SEC:
                    return t
            # Fall through if we couldn't find a clean spot

        # 3) Uniform away from ANY labeled event
        return self._uniform_sec_away_from_events(all_secs, duration, rnd)

    # -------- Preloader threads --------

    def _preload_worker(self):
        rnd = random.Random()
        while not self.stop_event.is_set():
            try:
                if self.frame_queue.full():
                    self.stop_event.wait(0.05)
                    continue

                # Weighted video choice (boost unseen & videos with more positive labels; perfect double)
                with self.ann_lock:
                    vid_path = self._choose_video_with_weights(rnd)
                    vid_name = self.video_name_by_path[vid_path]
                    # Snapshot events to avoid race during pick
                    all_secs_snapshot = [s for s, _ in self.annotations_sec.get(vid_name, [])]

                # Derive a target time with biasing rules
                try:
                    frame_count, fps = get_video_meta(vid_path)
                except Exception:
                    frame_count, fps = 0, 0.0

                if frame_count > 0 and fps > 0:
                    # try a few times to ensure MIN_GAP from any event after discretization
                    tries = 10
                    pil_img = None
                    ts_str = ""
                    for _ in range(tries):
                        target_sec = self._pick_biased_second(vid_name, frame_count, fps, rnd)
                        img, ts_str_candidate, ts_sec = read_frame_at_second(vid_path, target_sec)
                        # check gap after exact frame index -> seconds
                        nearest = min(abs(ts_sec - s) for s in all_secs_snapshot) if all_secs_snapshot else float("inf")
                        if nearest >= self.MIN_GAP_SEC:
                            pil_img = img
                            ts_str = ts_str_candidate
                            break
                    if pil_img is None:
                        # fallback uniform if biasing couldn't satisfy constraints
                        pil_img, ts_str, _ = get_random_frame(vid_path)
                else:
                    pil_img, ts_str, _ = get_random_frame(vid_path)

                payload = {
                    "video_path": vid_path,
                    "video_name": vid_name,
                    "ts_str": ts_str,
                    "pil_img": pil_img,
                }
                try:
                    self.frame_queue.put(payload, timeout=0.1)
                except queue.Full:
                    pass
            except Exception:
                continue

    def _start_preloader_workers(self):
        for i in range(self.NUM_WORKERS):
            t = threading.Thread(
                target=self._preload_worker,
                name=f"preloader-{i}",
                daemon=True
            )
            t.start()
            self.workers.append(t)

    # -------- UI frame consumption --------
    # NEW: try to synchronously show a frame near a given center, avoiding existing points by MIN_GAP_SEC.
    def _try_show_local_followup(self, video_name: str, center_ts_str: str) -> bool:
        """Returns True if we displayed a nearby frame, else False (caller can fall back)."""
        vid_path = self.video_path_by_name.get(video_name)
        if not vid_path:
            return False

        try:
            frame_count, fps = get_video_meta(vid_path)
        except Exception:
            return False

        duration = frame_count / max(fps, 1e-9)
        lo = max(0.02 * duration, 0.0)
        hi = max(0.0, 0.98 * duration)
        center = hhmmss_ms_to_seconds(center_ts_str)

        # Snapshot of existing annotated seconds for gap checks
        with self.ann_lock:
            all_secs = [s for s, _ in self.annotations_sec.get(video_name, [])]

        # Build a small set of nearby candidates just OUTSIDE the MIN_GAP_SEC ring
        radii = [
            self.MIN_GAP_SEC * 1.10,
            max(0.75, self.MIN_GAP_SEC * 1.50),
            max(1.25, self.MIN_GAP_SEC * 2.00),
            2.50,
            3.00,
        ]
        candidates = []
        for r in radii:
            candidates.append(center + r)
            candidates.append(center - r)

        # Keep candidates inside [lo, hi]
        candidates = [min(max(t, lo), hi) for t in candidates]

        # Try each candidate; after discrete frame read, re-check exact-gap
        for t in candidates:
            try:
                img, ts_str, ts_sec = read_frame_at_second(vid_path, t)
            except Exception:
                continue
            nearest = min(abs(ts_sec - s) for s in all_secs) if all_secs else float("inf")
            if nearest < self.MIN_GAP_SEC:
                continue
            # Looks good—display it directly.
            self.current_video_path = vid_path
            self.current_video_name = video_name
            self.current_timestamp_str = ts_str

            # Fit image into current window size
            self.root.update_idletasks()
            max_w = max(200, self.image_label.winfo_width() or 1000)
            max_h = max(200, (self.image_label.winfo_height() or 700))
            disp_img = resize_to_fit(img, max_w, max_h)
            self.current_imgtk = ImageTk.PhotoImage(disp_img)
            self.image_label.configure(image=self.current_imgtk)

            self.info_var.set(f"Video: {self.current_video_name}   |   Timestamp: {self.current_timestamp_str}   [near perfect]")
            return True

        return False

    def _show_next_preloaded_frame(self):
        # NEW: honor one-shot local follow-up after a "perfect"
        if self._pending_followup is not None:
            video_name, center_ts_str = self._pending_followup
            # Clear the request first to ensure one-shot behavior even if it fails
            self._pending_followup = None
            if self._try_show_local_followup(video_name, center_ts_str):
                return  # successfully displayed a local frame; we're done

        try:
            item = self.frame_queue.get_nowait()
        except queue.Empty:
            self.info_var.set("Loading next frame…")
            self.root.after(50, self._show_next_preloaded_frame)
            return

        self.current_video_path = item["video_path"]
        self.current_video_name = item["video_name"]
        self.current_timestamp_str = item["ts_str"]

        # Fit image into current window size
        self.root.update_idletasks()
        max_w = max(200, self.image_label.winfo_width() or 1000)
        max_h = max(200, (self.image_label.winfo_height() or 700))

        disp_img = resize_to_fit(item["pil_img"], max_w, max_h)
        self.current_imgtk = ImageTk.PhotoImage(disp_img)
        self.image_label.configure(image=self.current_imgtk)

        self.info_var.set(f"Video: {self.current_video_name}   |   Timestamp: {self.current_timestamp_str}")

    def _choose_video_with_weights(self, rnd: random.Random) -> str:
        weights = []
        for p in self.videos:
            name = self.video_name_by_path[p]
            w = self._video_pick_weight(name)
            weights.append(max(w, 0.001))

        if hasattr(rnd, "choices"):
            return rnd.choices(self.videos, weights=weights, k=1)[0]

        total = sum(weights)
        r = rnd.random() * total
        acc = 0.0
        for p, w in zip(self.videos, weights):
            acc += w
            if r <= acc:
                return p
        return self.videos[-1]

    def record_and_next(self, label: str):
        if not self.current_video_name or not self.current_timestamp_str:
            self._show_next_preloaded_frame()
            return

        ts_str = self.current_timestamp_str
        lab = _normalize_label(label)
        if not lab:
            self._show_next_preloaded_frame()
            return
        vid_name = self.current_video_name

        # Update canonical map (thread-safe): last-write-wins, no duplicates
        with self.ann_lock:
            prev = self.ann_map.setdefault(vid_name, {}).get(ts_str)
            if prev == lab:
                pass
            else:
                self.ann_map[vid_name][ts_str] = lab
                # Rebuild in-memory caches
                self._rebuild_secondary_caches()
                # Persist atomically
                try:
                    write_canonical_annotations_atomic(self.annotation_file, self.ann_map)
                except Exception as e:
                    messagebox.showwarning("Save warning", f"Could not write annotations:\n{e}")
        # NEW: if this was a "perfect", request a one-shot local follow-up near this timestamp
        if lab == "perfect":
            self._pending_followup = (vid_name, ts_str)
        # Next frame (instant if queue has items)
        self._show_next_preloaded_frame()

    def on_close(self):
        try:
            pass
        except Exception:
            pass
        self.stop_event.set()
        for t in self.workers:
            t.join(timeout=0.2)
        self.root.destroy()

# ----------------------- Main -----------------------

def main():
    # >>>>>>>> SET YOUR VIDEO FOLDER HERE (Windows path) <<<<<<<<
    VIDEO_DIR = r"D:\NewFolder(3)\videoProject\Charlote_Videos"
    # -----------------------------------------------------------

    if not os.path.isdir(VIDEO_DIR):
        raise SystemExit(f"Video directory does not exist: {VIDEO_DIR}")

    root = tk.Tk()
    app = AnnotatorApp(root, VIDEO_DIR)
    try:
        root.mainloop()
    finally:
        pass

if __name__ == "__main__":
    main()

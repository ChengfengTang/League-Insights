"""
Live minimap tracker: capture a screen region (minimap), detect jungler icons,
and record (x, y, timestep) every second. Rough base — region is fixed; UI later.
"""

import json
import os
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# Optional: screen capture and vision (install: pip install mss opencv-python-headless numpy)
try:
    import mss
    import numpy as np
    import cv2
    HAS_CAPTURE = True
except ImportError:
    HAS_CAPTURE = False

# ---------------------------------------------------------------------------
# Hardcoded config (replace with UI / config later)
# ---------------------------------------------------------------------------
JUNGLER_1 = "LeeSin"   # Ally or first jungler
JUNGLER_2 = "Graves"   # Enemy or second jungler

# Fixed minimap region: (left, top, width, height) in screen pixels.
# Default: bottom-left 256x256 — adjust for your resolution / LoL window.
MINIMAP_REGION = {
    "left": 20,
    "top": 800,
    "width": 256,
    "height": 256,
}

# Sample every N seconds
SAMPLE_INTERVAL_SEC = 1.0


@dataclass
class JunglerSnapshot:
    """One recorded position for a jungler at a timestep."""
    jungler_id: int
    jungler_name: str
    x: int
    y: int
    t_sec: float

    def to_dict(self):
        return {
            "jungler_id": self.jungler_id,
            "jungler_name": self.jungler_name,
            "x": self.x,
            "y": self.y,
            "t_sec": self.t_sec,
        }


@dataclass
class TrackerState:
    """Gathered data and running state."""
    jungler_1_name: str
    jungler_2_name: str
    records: List[JunglerSnapshot] = field(default_factory=list)
    start_time: Optional[float] = None

    def add(self, snap: JunglerSnapshot):
        self.records.append(snap)

    def get_records(self) -> List[dict]:
        return [r.to_dict() for r in self.records]


def capture_region(region: dict) -> Optional["np.ndarray"]:
    """Capture the given screen region; returns BGR numpy array or None."""
    if not HAS_CAPTURE:
        return None
    with mss.mss() as sct:
        shot = sct.grab(region)
        # mss returns BGRA; convert to BGR for OpenCV
        frame = np.array(shot)[:, :, :3]
        return frame
    return None


def find_icon_positions(frame: "np.ndarray", max_icons: int = 2) -> List[Tuple[int, int]]:
    """
    Detect up to max_icons icon-like positions on the minimap (rough base).
    Uses red/enemy-colored blobs; replace with template matching per champ later.
    Returns list of (x, y) in region pixel coordinates.
    """
    if not HAS_CAPTURE or frame is None or frame.size == 0:
        return []

    # Convert BGR -> HSV and look for red (enemy) hue range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red wraps in HSV: low red and high red
    low_red1 = np.array([0, 100, 100])
    high_red1 = np.array([10, 255, 255])
    low_red2 = np.array([160, 100, 100])
    high_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, low_red1, high_red1)
    mask2 = cv2.inRange(hsv, low_red2, high_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Optional: also detect blue (ally) so we have two "icons" for two junglers
    low_blue = np.array([100, 100, 100])
    high_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    combined = cv2.bitwise_or(red_mask, blue_mask)

    # Find contours and take centroids of small blobs (icon-sized)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    positions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10 or area > 800:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        positions.append((cx, cy))

    # Sort by y then x for stable ordering; take up to max_icons
    positions.sort(key=lambda p: (p[1], p[0]))
    return positions[:max_icons]


def run_tracker(
    region: Optional[dict] = None,
    interval_sec: float = SAMPLE_INTERVAL_SEC,
    stop_after_sec: Optional[float] = None,
) -> TrackerState:
    """
    Run the minimap tracker: capture region every interval_sec, detect icons,
    assign to jungler 0 and 1, record (x, y, t_sec). Runs until keyboard interrupt
    or stop_after_sec.
    """
    region = region or MINIMAP_REGION
    state = TrackerState(
        jungler_1_name=JUNGLER_1,
        jungler_2_name=JUNGLER_2,
    )
    state.start_time = time.perf_counter()
    names = [JUNGLER_1, JUNGLER_2]

    print(f"Live minimap tracker — junglers: {JUNGLER_1}, {JUNGLER_2}")
    print(f"Region: {region}, interval: {interval_sec}s. Stop with Ctrl+C.")
    if not HAS_CAPTURE:
        print("Warning: mss/opencv/numpy not installed. Install with: pip install mss opencv-python-headless numpy")
        return state

    try:
        while True:
            t = time.perf_counter() - state.start_time
            if stop_after_sec is not None and t >= stop_after_sec:
                break

            frame = capture_region(region)
            positions = find_icon_positions(frame, max_icons=2) if frame is not None else []

            for i, (px, py) in enumerate(positions):
                if i >= 2:
                    break
                snap = JunglerSnapshot(
                    jungler_id=i,
                    jungler_name=names[i],
                    x=px,
                    y=py,
                    t_sec=round(t, 1),
                )
                state.add(snap)
                print(f"  t={snap.t_sec}s  {snap.jungler_name}: ({snap.x}, {snap.y})")

            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"Gathered {len(state.records)} records.")
    return state


def save_records(state: TrackerState, path: Optional[str] = None) -> str:
    """Save gathered records to JSON. Returns path used."""
    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "Live", "live_records.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "jungler_1": state.jungler_1_name,
        "jungler_2": state.jungler_2_name,
        "records": state.get_records(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    # Hardcoded junglers (already set at top)
    # JUNGLER_1 = "LeeSin"
    # JUNGLER_2 = "Graves"

    state = run_tracker(
        region=MINIMAP_REGION,
        interval_sec=SAMPLE_INTERVAL_SEC,
        stop_after_sec=None,
    )

    # Save gathered data for later use (e.g. feed to predictor)
    records = state.get_records()
    if records:
        out_path = save_records(state)
        print("Sample records:", records[:5])
        print(f"Saved all to {out_path}")
    return state


if __name__ == "__main__":
    main()

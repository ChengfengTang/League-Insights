"""
Minimap capture and coordinate mapping.
Captures a screen region (the in-game minimap), detects enemy (red) champion blobs,
and maps pixel positions to League in-game coordinates (same scale as timeline/match data).
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Optional

try:
    import mss
    import mss.tools
except ImportError:
    mss = None

try:
    import cv2
except ImportError:
    cv2 = None

from .config import (
    MAP_MIN_X,
    MAP_MIN_Y,
    MAP_MAX_X,
    MAP_MAX_Y,
    DEFAULT_MINIMAP,
    ENEMY_RED_RGB_LOW,
    ENEMY_RED_RGB_HIGH,
    MIN_BLOB_AREA,
)


def pixel_to_game(
    px: float,
    py: float,
    minimap_width: int,
    minimap_height: int,
) -> Tuple[float, float]:
    """
    Map minimap pixel (px, py) to in-game coordinates (x, y).
    Top-left of minimap = (0, 0) in game; bottom-right = (MAP_MAX_X, MAP_MAX_Y).
    Matches the coordinate system used in Riot API timeline positions.
    """
    gx = MAP_MIN_X + (px / max(1, minimap_width)) * (MAP_MAX_X - MAP_MIN_X)
    gy = MAP_MIN_Y + (py / max(1, minimap_height)) * (MAP_MAX_Y - MAP_MIN_Y)
    return (gx, gy)


def capture_region(region: Optional[Dict[str, int]] = None) -> Optional[np.ndarray]:
    """
    Capture the minimap region as numpy array (RGB, shape HxWx3).
    region: dict with keys left, top, width, height. Defaults to config DEFAULT_MINIMAP.
    """
    if mss is None:
        raise ImportError("Install mss: pip install mss")
    region = region or DEFAULT_MINIMAP
    with mss.mss() as sct:
        mon = sct.monitors[0]
        # Clip to monitor
        x = max(0, region["left"])
        y = max(0, region["top"])
        w = region["width"]
        h = region["height"]
        if x + w > mon["width"] or y + h > mon["height"]:
            return None
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
        # mss returns BGRA
        img = np.array(shot)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img[:, :, ::-1])
        return img


def detect_enemy_blobs(
    rgb: np.ndarray,
    low_rgb: Tuple[int, int, int] = ENEMY_RED_RGB_LOW,
    high_rgb: Tuple[int, int, int] = ENEMY_RED_RGB_HIGH,
    min_area: int = MIN_BLOB_AREA,
) -> List[Tuple[float, float]]:
    """
    Find red (enemy) blobs in the minimap image. Returns list of (px, py) pixel centroids.
    """
    if rgb is None or rgb.size == 0:
        return []
    if cv2 is None:
        return _detect_blobs_numpy(rgb, low_rgb, high_rgb, min_area)
    low = np.array(low_rgb, dtype=np.uint8)
    high = np.array(high_rgb, dtype=np.uint8)
    mask = cv2.inRange(rgb, low, high)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    centroids = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroids.append((float(cx), float(cy)))
    return centroids


def _detect_blobs_numpy(
    rgb: np.ndarray,
    low_rgb: Tuple[int, int, int],
    high_rgb: Tuple[int, int, int],
    min_area: int,
) -> List[Tuple[float, float]]:
    """Fallback blob detection without OpenCV: mask red, then simple connected regions."""
    low = np.array(low_rgb)
    high = np.array(high_rgb)
    mask = np.all((rgb >= low) & (rgb <= high), axis=2).astype(np.uint8)
    # Label connected components with a simple flood-fill style grouping by distance
    h, w = mask.shape
    visited = np.zeros_like(mask)
    centroids = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            xs, ys = [], []
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w or visited[cy, cx] or mask[cy, cx] == 0:
                    continue
                visited[cy, cx] = 1
                xs.append(cx)
                ys.append(cy)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cy + dy, cx + dx))
            if len(xs) >= min_area:
                centroids.append((np.mean(xs), np.mean(ys)))
    return centroids


def get_enemy_positions_game(
    region: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[np.ndarray], List[Tuple[float, float]]]:
    """
    Capture minimap, detect enemy blobs, return (image, list of (game_x, game_y)).
    Image is the RGB capture for debugging/display; positions are in game coordinates.
    """
    img = capture_region(region)
    if img is None:
        return None, []
    blobs_px = detect_enemy_blobs(img)
    h_img, w_img = img.shape[0], img.shape[1]
    positions = [pixel_to_game(px, py, w_img, h_img) for (px, py) in blobs_px]
    return img, positions

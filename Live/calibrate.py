"""
One-shot capture of the configured minimap region. Use to verify region (left, top, width, height)
matches your game's minimap. Save the image and adjust Live/config.py or use --left/--top/--width/--height.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Live.config import DEFAULT_MINIMAP
from Live.minimap import capture_region

if __name__ == "__main__":
    region = DEFAULT_MINIMAP
    img = capture_region(region)
    if img is None:
        print("Capture failed. Check region bounds.")
        sys.exit(1)
    import cv2
    out = Path(__file__).parent / "minimap_calibrate.png"
    cv2.imwrite(str(out), img[:, :, ::-1])  # OpenCV expects BGR
    print(f"Saved {out}. Adjust Live/config.py DEFAULT_MINIMAP if this isn't your minimap.")

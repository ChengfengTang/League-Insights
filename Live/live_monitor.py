"""
Live minimap monitor: capture the game minimap, detect enemy positions in game coordinates,
record them for training, and optionally run ML predictions (same coordinate system as timeline data).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Ensure project root and Predict are on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from same package (Live)
from Live.minimap import get_enemy_positions_game, capture_region
from Live.config import DEFAULT_MINIMAP, CAPTURE_INTERVAL, CAPTURES_FILE


def _ensure_live_captures_path() -> str:
    out = os.path.join(project_root, "Live", CAPTURES_FILE)
    return out


def load_captures(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or _ensure_live_captures_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_captures(records: List[Dict[str, Any]], path: Optional[str] = None) -> None:
    path = path or _ensure_live_captures_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def run_monitor(
    region: Optional[Dict[str, int]] = None,
    interval: float = CAPTURE_INTERVAL,
    use_predictor: bool = True,
    predictor_type: str = "lstm",
    game_time_sec: Optional[float] = None,
) -> None:
    """
    Run the live minimap monitor loop.
    - Captures the minimap region, detects enemy (red) blobs, maps to game (x, y).
    - Appends each capture to history and saves to live_captures.json.
    - If use_predictor and a trained model exists, prints predicted jungler position.
    - region: { left, top, width, height } for the minimap on screen (default from config).
    - game_time_sec: if set, used as current game time for predictor; else elapsed since start.
    """
    region = region or DEFAULT_MINIMAP
    start_time = time.time()
    history: List[Dict[str, Any]] = []
    predictor = None
    lstm_seq_len = 30

    if use_predictor:
        try:
            if predictor_type == "lstm":
                from Predict.LSTMpredict import LSTMJunglerPredictor
                predictor = LSTMJunglerPredictor(seq_len=lstm_seq_len)
                predictor.load_model()
            else:
                from Predict.TDpredict import JunglerPredictor
                predictor = JunglerPredictor(model_type="gradient_boosting", use_categories=False)
                predictor.load_model(category=None)
        except Exception as e:
            print(f"⚠️  Could not load predictor: {e}. Continuing without predictions.")
            predictor = None

    print("Live minimap monitor started. Minimap coordinates match League timeline data (0–14820 x, 0–14881 y).")
    print("Press Ctrl+C to stop and save captures.")
    print()

    try:
        while True:
            t = time.time()
            elapsed = t - start_time
            game_time = game_time_sec if game_time_sec is not None else elapsed

            img, positions = get_enemy_positions_game(region)
            if img is not None and positions:
                record = {
                    "timestamp_sec": round(elapsed, 2),
                    "game_time_sec": round(game_time, 2),
                    "positions": [{"x": round(x, 2), "y": round(y, 2)} for (x, y) in positions],
                }
                history.append(record)
                print(f"[{elapsed:.1f}s] Detected {len(positions)} position(s): {[(round(x,0), round(y,0)) for (x,y) in positions]}")

                if predictor and positions:
                    last_x, last_y = positions[-1]
                    try:
                        if predictor_type == "lstm":
                            if len(history) >= lstm_seq_len:
                                # Build sequence: each row [x, y, time, game_minutes, level, cs, gold]
                                seq = []
                                for r in history[-lstm_seq_len:]:
                                    gt = r["game_time_sec"]
                                    pos = r["positions"][0] if r["positions"] else {"x": last_x, "y": last_y}
                                    seq.append([
                                        pos["x"], pos["y"], gt, gt / 60.0, 1, 0, 0
                                    ])
                                import numpy as np
                                pred_x, pred_y = predictor.predict(np.array(seq, dtype=np.float32))
                                print(f"         LSTM prediction (30s ahead): ({pred_x:.0f}, {pred_y:.0f})")
                        else:
                            pred_x, pred_y = predictor.predict(
                                last_x, last_y, game_time,
                                game_minutes=game_time / 60.0, level=1, cs=0, gold=0
                            )
                            print(f"         Prediction (30s ahead): ({pred_x:.0f}, {pred_y:.0f})")
                    except Exception as e:
                        print(f"         Prediction error: {e}")
            else:
                print(f"[{elapsed:.1f}s] No enemy positions detected.")

            sleep_for = max(0, interval - (time.time() - t))
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if history:
            path = _ensure_live_captures_path()
            save_captures(history, path)
            print(f"Saved {len(history)} captures to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live minimap monitor (game coords = timeline coords)")
    parser.add_argument("--no-predict", action="store_true", help="Disable ML prediction")
    parser.add_argument("--predictor", choices=["lstm", "tree"], default="lstm", help="Which model to use")
    parser.add_argument("--interval", type=float, default=CAPTURE_INTERVAL, help="Capture interval in seconds")
    parser.add_argument("--left", type=int, default=None, help="Minimap region left (px)")
    parser.add_argument("--top", type=int, default=None, help="Minimap region top (px)")
    parser.add_argument("--width", type=int, default=None, help="Minimap width (px)")
    parser.add_argument("--height", type=int, default=None, help="Minimap height (px)")
    args = parser.parse_args()

    region = None
    if any([args.left is not None, args.top is not None, args.width is not None, args.height is not None]):
        region = {
            "left": args.left or DEFAULT_MINIMAP["left"],
            "top": args.top or DEFAULT_MINIMAP["top"],
            "width": args.width or DEFAULT_MINIMAP["width"],
            "height": args.height or DEFAULT_MINIMAP["height"],
        }

    run_monitor(
        region=region,
        interval=args.interval,
        use_predictor=not args.no_predict,
        predictor_type=args.predictor,
    )

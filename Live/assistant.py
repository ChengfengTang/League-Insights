"""
Live in-game AI assistant: text chat with an AI that sees minimap data, game time,
and our trained jungler prediction. Tell the AI what you're playing; it uses live
context to help with jungle tracking and advice.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Live.minimap import get_enemy_positions_game
from Live.config import DEFAULT_MINIMAP, CAPTURE_INTERVAL

# Shared state (updated by capture thread, read by main thread)
_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "game_time_sec": 0.0,
    "start_time": 0.0,
    "last_positions": [],
    "last_prediction": None,
    "position_history": [],
}


def _capture_loop(region: Dict[str, int], use_predictor: bool, predictor_type: str):
    """Background thread: capture minimap, update shared state, optional prediction."""
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
        except Exception:
            predictor = None

    start = time.time()
    while True:
        t = time.time()
        elapsed = t - start
        img, positions = get_enemy_positions_game(region)
        pred = None
        if predictor and positions:
            last_x, last_y = positions[-1]
            try:
                if predictor_type == "lstm":
                    hist = _state.get("position_history", [])
                    current_rec = {"game_time_sec": elapsed, "positions": [{"x": last_x, "y": last_y}]}
                    combined = (hist + [current_rec])[-lstm_seq_len:]
                    if len(combined) >= lstm_seq_len:
                        import numpy as np
                        seq = []
                        for r in combined[-lstm_seq_len:]:
                            gt = r.get("game_time_sec", 0)
                            pos = r["positions"][0] if r["positions"] else {"x": last_x, "y": last_y}
                            seq.append([pos["x"], pos["y"], gt, gt / 60.0, 1, 0, 0])
                        pred = predictor.predict(np.array(seq, dtype=np.float32))
                else:
                    pred = predictor.predict(last_x, last_y, elapsed, game_minutes=elapsed / 60.0, level=1, cs=0, gold=0)
            except Exception:
                pred = None

        with _state_lock:
            _state["game_time_sec"] = elapsed
            _state["start_time"] = start
            if img is not None and positions:
                _state["last_positions"] = [(round(x, 1), round(y, 1)) for (x, y) in positions]
                rec = {
                    "game_time_sec": round(elapsed, 2),
                    "positions": [{"x": round(x, 2), "y": round(y, 2)} for (x, y) in positions],
                }
                _state["position_history"] = (_state.get("position_history") or [])[-50:] + [rec]
            else:
                _state["last_positions"] = _state.get("last_positions", [])
            _state["last_prediction"] = (round(pred[0], 1), round(pred[1], 1)) if pred else None

        time.sleep(max(0, CAPTURE_INTERVAL - (time.time() - t)))


def _build_context() -> str:
    """Build a short context string for the LLM from current shared state."""
    with _state_lock:
        gt = _state.get("game_time_sec", 0)
        pos = _state.get("last_positions", [])
        pred = _state.get("last_prediction")
    mins = int(gt // 60)
    secs = int(gt % 60)
    time_str = f"{mins}:{secs:02d}"
    lines = [
        f"Game time: {time_str} (minutes:seconds).",
        f"Enemy positions seen on minimap (game coordinates, last capture): {pos if pos else 'none'}.",
        f"Model prediction (enemy jungler position ~30s ahead): {pred}." if pred else "No prediction yet (need more minimap data).",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an in-game League of Legends assistant. You have access to live minimap data and a trained model that predicts where the enemy jungler will be in about 30 seconds.

Context you receive includes:
- Game time (minutes:seconds).
- Enemy positions recently seen on the minimap (game coordinates: x,y from 0 to ~14820).
- A model prediction for enemy jungler location ~30 seconds ahead (when available).
- What the user says (e.g. "I'm playing mid Ahri", "where is their jg?").

Help the user with:
- Where the enemy jungler likely is or is heading.
- When to expect ganks, where to ward, and when to play safe.
- Short, actionable advice. You don't see the full map—only what appears on the minimap and the prediction.

Keep replies concise and in-game friendly (a few sentences). Use the coordinates to describe map areas (e.g. "top side", "near dragon", "their jungle") when helpful."""


def _call_llm(user_message: str, context: str, history: List[Dict[str, str]]) -> str:
    """Call OpenAI-compatible chat API. Returns assistant reply or error message."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key and not base_url:
        return (
            "No LLM configured. Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL for a local model like Ollama). "
            "I'm still watching the minimap; you can run with an API key to get AI replies."
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or "ollama", base_url=base_url or "http://localhost:11434/v1")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current live context:\n{context}\n\nUser message: {user_message}"},
        ]
        if history:
            for h in history[-6:]:
                messages.insert(-1, {"role": h["role"], "content": h["content"]})
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=400)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error calling AI: {e}"


def run_assistant(
    region: Optional[Dict[str, int]] = None,
    use_predictor: bool = True,
    predictor_type: str = "lstm",
):
    """Run the text-based AI assistant: background minimap + predictor, foreground chat."""
    region = region or DEFAULT_MINIMAP
    daemon = threading.Thread(
        target=_capture_loop,
        args=(region, use_predictor, predictor_type),
        daemon=True,
    )
    daemon.start()
    time.sleep(0.5)

    print("Live in-game AI assistant")
    print("Minimap and (if available) jungler prediction are running in the background.")
    print("Type your message and press Enter. Say e.g. 'I'm playing mid Ahri' or 'Where is their jg?'")
    print("Commands: quit / exit = stop. clear = clear conversation context.")
    print()
    history: List[Dict[str, str]] = []

    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if line.lower() == "clear":
            history.clear()
            print("Context cleared.")
            continue

        context = _build_context()
        reply = _call_llm(line, context, history)
        print("AI:", reply)
        print()
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Live in-game AI assistant (text chat)")
    p.add_argument("--no-predict", action="store_true", help="Disable jungler prediction model")
    p.add_argument("--predictor", choices=["lstm", "tree"], default="lstm")
    p.add_argument("--left", type=int, default=None, help="Minimap region left (px)")
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    args = p.parse_args()
    region = None
    if any([args.left is not None, args.top is not None, args.width is not None, args.height is not None]):
        region = {
            "left": args.left if args.left is not None else DEFAULT_MINIMAP["left"],
            "top": args.top if args.top is not None else DEFAULT_MINIMAP["top"],
            "width": args.width if args.width is not None else DEFAULT_MINIMAP["width"],
            "height": args.height if args.height is not None else DEFAULT_MINIMAP["height"],
        }
    run_assistant(region=region, use_predictor=not args.no_predict, predictor_type=args.predictor)

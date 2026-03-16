"""
League of Legends Enemy Jungler Location Predictor (LSTM)
Uses a sequence of past frames (position, time, level, cs, gold) to predict
enemy jungler location 30 seconds ahead via a PyTorch LSTM.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union

try:
    from champCategory import get_champion_category
except ImportError:
    get_champion_category = None

# Project paths (same as TDpredict)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
timelines_dir = os.path.join(project_root, "timelines")
matches_dir = os.path.join(project_root, "matches")
models_dir = os.path.join(project_root, "Predict", "models")
os.makedirs(models_dir, exist_ok=True)

FEATURE_COLS = ["x", "y", "time", "game_minutes", "level", "cs", "gold"]
FUTURE_SECONDS = 30


def extract_features(
    timeline_data: Dict,
    match_data: Dict,
    participant_id: int,
) -> pd.DataFrame:
    """Extract per-frame features for one participant (enemy jungler). Same logic as TDpredict."""
    features = []
    frames = timeline_data["info"]["frames"]
    for frame in frames:
        timestamp_ms = frame["timestamp"]
        timestamp_sec = timestamp_ms / 1000.0
        game_minutes = timestamp_sec / 60.0
        participant_frame = frame["participantFrames"].get(str(participant_id))
        if not participant_frame or "position" not in participant_frame:
            continue
        pos = participant_frame["position"]
        x, y = pos["x"], pos["y"]
        level = participant_frame.get("level", 1)
        cs = participant_frame.get("minionsKilled", 0) + participant_frame.get("jungleMinionsKilled", 0)
        gold = participant_frame.get("currentGold", 0)
        features.append({
            "x": x, "y": y, "time": timestamp_sec, "game_minutes": game_minutes,
            "level": level, "cs": cs, "gold": gold,
        })
    return pd.DataFrame(features)


def build_sequences(
    features_df: pd.DataFrame,
    seq_len: int,
    future_seconds: int = FUTURE_SECONDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) where X has shape (n_samples, seq_len, n_features) and y has shape (n_samples, 2).
    Each sample is a contiguous window of seq_len frames; target is (future_x, future_y) at
    future_seconds after the last frame in the window.
    """
    X_list, y_list = [], []
    for i in range(len(features_df) - seq_len):
        window = features_df.iloc[i : i + seq_len][FEATURE_COLS].values.astype(np.float32)
        last_time = features_df.iloc[i + seq_len - 1]["time"]
        future_time = last_time + future_seconds
        future_rows = features_df[features_df["time"] >= future_time]
        if len(future_rows) == 0:
            continue
        future_row = future_rows.iloc[0]
        target = np.array([future_row["x"], future_row["y"]], dtype=np.float32)
        X_list.append(window)
        y_list.append(target)
    if len(X_list) == 0:
        return np.array([]), np.array([])
    return np.stack(X_list), np.stack(y_list)


class JunglerSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: StandardScaler):
        self.X = torch.from_numpy(scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape))
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMJunglerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return out


class LSTMJunglerPredictor:
    """
    LSTM-based predictor: uses a sequence of past frames to predict future (x, y).
    """

    def __init__(
        self,
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        device: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = len(FEATURE_COLS)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model: Optional[LSTMJunglerModel] = None

    def _create_model(self) -> LSTMJunglerModel:
        return LSTMJunglerModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

    def prepare_training_data(
        self,
        match_files: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_X, all_y = [], []
        for match_id in match_files:
            timeline_path = os.path.join(timelines_dir, f"{match_id}_timeline.json")
            match_path = os.path.join(matches_dir, f"{match_id}.json")
            if not os.path.exists(timeline_path) or not os.path.exists(match_path):
                print(f"⚠️  Skipping {match_id}: files not found")
                continue
            try:
                with open(timeline_path) as f:
                    timeline_data = json.load(f)
                with open(match_path) as f:
                    match_data = json.load(f)
                enemy_jungler_id = 6
                features_df = extract_features(timeline_data, match_data, enemy_jungler_id)
                if len(features_df) < self.seq_len + 1:
                    continue
                X, y = build_sequences(features_df, self.seq_len)
                if len(X) == 0:
                    continue
                all_X.append(X)
                all_y.append(y)
            except Exception as e:
                print(f"❌ Error processing {match_id}: {e}")
        if len(all_X) == 0:
            raise ValueError("No training data could be loaded.")
        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    def train(
        self,
        match_files: List[str],
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        save_model: bool = True,
    ) -> None:
        print(f"📊 Loading training data from {len(match_files)} matches...")
        X, y = self.prepare_training_data(match_files)
        print(f"✅ Built {len(X)} sequences (seq_len={self.seq_len})")

        # Fit scaler on all sequences (reshape to (N*seq_len, n_features))
        n, sl, f = X.shape
        self.scaler.fit(X.reshape(-1, f))

        dataset = JunglerSequenceDataset(X, y, self.scaler)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = self._create_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.4f}")

        # Eval
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(
                self.scaler.transform(X.reshape(-1, f)).reshape(X.shape)
            ).to(self.device)
            pred = self.model(X_t)
            mse = nn.functional.mse_loss(pred, torch.from_numpy(y).to(self.device)).item()
            mae = nn.functional.l1_loss(pred, torch.from_numpy(y).to(self.device)).item()
        print(f"✅ Train metrics: MSE={mse:.2f}, MAE={mae:.2f}")

        if save_model:
            path = os.path.join(models_dir, "lstm_jungler.pt")
            torch.save({
                "state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler.mean_.tolist(),
                "scaler_scale": self.scaler.scale_.tolist(),
                "seq_len": self.seq_len,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            }, path)
            print(f"💾 Saved model to {path}")

    def predict(
        self,
        sequence: Union[np.ndarray, List],
    ) -> Tuple[float, float]:
        """
        Predict (future_x, future_y) from a sequence of frames.
        sequence: shape (seq_len, 7) with columns [x, y, time, game_minutes, level, cs, gold].
        """
        if self.model is None:
            raise ValueError("No trained model. Train or load a model first.")
        X = np.asarray(sequence, dtype=np.float32)
        if X.shape != (self.seq_len, self.input_size):
            raise ValueError(f"Expected sequence shape ({self.seq_len}, {self.input_size}), got {X.shape}")
        X = self.scaler.transform(X)
        X = torch.from_numpy(X).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).cpu().numpy()[0]
        return float(pred[0]), float(pred[1])

    def load_model(self) -> None:
        path = os.path.join(models_dir, "lstm_jungler.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.seq_len = ckpt["seq_len"]
        self.input_size = ckpt["input_size"]
        self.hidden_size = ckpt["hidden_size"]
        self.num_layers = ckpt["num_layers"]
        self.scaler.mean_ = np.array(ckpt["scaler_mean"], dtype=np.float64)
        self.scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
        self.model = self._create_model()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(f"✅ Loaded LSTM model from {path}")


def get_available_matches() -> List[str]:
    match_ids = []
    for filename in os.listdir(timelines_dir):
        if filename.endswith("_timeline.json"):
            match_ids.append(filename.replace("_timeline.json", ""))
    return match_ids


if __name__ == "__main__":
    print("🧠 LSTM Enemy Jungler Location Predictor")
    print("=" * 50)
    predictor = LSTMJunglerPredictor(seq_len=30, hidden_size=64, num_layers=2)
    match_files = get_available_matches()
    print(f"\n📁 Found {len(match_files)} match files")
    if len(match_files) == 0:
        print("⚠️  No match files found. Run fetchdata.py first.")
    else:
        training_matches = match_files[:10]
        print(f"🎯 Training on {len(training_matches)} matches...")
        try:
            predictor.train(training_matches, epochs=50, batch_size=64, save_model=True)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        # Example: predict from a random sequence (in practice you'd pass last seq_len frames)
        print("\n🔮 Example: load first match and predict from first sequence...")
        try:
            timeline_path = os.path.join(timelines_dir, f"{match_files[0]}_timeline.json")
            match_path = os.path.join(matches_dir, f"{match_files[0]}.json")
            with open(timeline_path) as f:
                tl = json.load(f)
            with open(match_path) as f:
                meta = json.load(f)
            features_df = extract_features(tl, meta, 6)
            X, y = build_sequences(features_df, predictor.seq_len)
            if len(X) > 0:
                pred_x, pred_y = predictor.predict(X[0])
                print(f"   Predicted: ({pred_x:.2f}, {pred_y:.2f}) (true: {y[0][0]:.2f}, {y[0][1]:.2f})")
        except Exception as e:
            print(f"   Example prediction failed: {e}")

"""
Train a GRU-based region classifier from saved `Train/Input/<Champion>/`.

Run `Train/loadData.py` first to export one champion dataset under `Train/Input/`.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]

TORCH_IMPORT_ERROR: Optional[Exception] = None
try:
    import torch  # pyright: ignore[reportMissingImports]
    from torch import nn  # pyright: ignore[reportMissingImports]
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    TORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment,misc]
    IterableDataset = object  # type: ignore[assignment,misc]

try:
    from loadData import (
        CHAMPION,
        INPUT_ROOT,
        PROGRESS_PRINT_EVERY_COUNT,
        PROGRESS_PRINT_EVERY_SECONDS,
        REGION_NAMES,
        ROW_TYPES,
        WINDOW_SIZE as DEFAULT_WINDOW_SIZE,
        champion_input_dir,
        format_progress_bar,
        iter_saved_metadata,
        load_saved_feature_schema,
        load_saved_summary,
        resolve_saved_shard_path,
        saved_dataset_paths,
    )
except ModuleNotFoundError:
    from Train.loadData import (
        CHAMPION,
        INPUT_ROOT,
        PROGRESS_PRINT_EVERY_COUNT,
        PROGRESS_PRINT_EVERY_SECONDS,
        REGION_NAMES,
        ROW_TYPES,
        WINDOW_SIZE as DEFAULT_WINDOW_SIZE,
        champion_input_dir,
        format_progress_bar,
        iter_saved_metadata,
        load_saved_feature_schema,
        load_saved_summary,
        resolve_saved_shard_path,
        saved_dataset_paths,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_ROOT = INPUT_ROOT
MODEL_CHAMPION = CHAMPION
WINDOW_SIZE = max(32, DEFAULT_WINDOW_SIZE)
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
MAX_TRAIN_SAMPLES: Optional[int] = None
MAX_VAL_SAMPLES = 500_000
MAX_TEST_SAMPLES = 500_000
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 256
EMBED_DIM = 16
GRU_LAYERS = 2
DROPOUT = 0.2
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 4
LR_SCHEDULER_PATIENCE = 1
LR_SCHEDULER_FACTOR = 0.5
MAX_NORMALIZER_FILES = 20_000
TRAIN_PROGRESS_PRINT_EVERY_COUNT = 25_000
TRAIN_PROGRESS_PRINT_EVERY_SECONDS = 30.0
NUM_WORKERS = 0
MODEL_OUT = REPO_ROOT / "Train" / "models" / "region_classifier.pt"
SUMMARY_OUT = REPO_ROOT / "Train" / "models" / "region_classifier.summary.json"


def _require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch is required for `Train/train.py`. "
            "Install it first with `python3 -m pip install torch`."
        ) from TORCH_IMPORT_ERROR


def _default_device() -> str:
    _require_torch()
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _print_stage_progress(
    label: str,
    done: int,
    total: int,
    *,
    count_every: Optional[int] = None,
    seconds_every: Optional[float] = None,
    force: bool = False,
    state: Dict[str, float],
) -> None:
    now = time.time()
    last_count = int(state.get("count", 0))
    last_time = float(state.get("time", 0.0))
    count_threshold = PROGRESS_PRINT_EVERY_COUNT if count_every is None else max(1, int(count_every))
    time_threshold = PROGRESS_PRINT_EVERY_SECONDS if seconds_every is None else max(0.0, float(seconds_every))
    should_print = (
        force
        or done == total
        or done - last_count >= count_threshold
        or now - last_time >= time_threshold
    )
    if should_print:
        print(f"[train] {label} {format_progress_bar(done, total)}", flush=True)
        state["count"] = float(done)
        state["time"] = now


def _group_train_test_match_ids(
    match_ids: Sequence[str],
    test_size: float,
    random_state: int,
) -> Tuple[List[str], List[str]]:
    unique_groups = sorted(set(match_ids))
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 unique matchIds for a train/test split.")

    rng = random.Random(random_state)
    rng.shuffle(unique_groups)

    test_group_count = int(round(len(unique_groups) * test_size))
    test_group_count = max(1, min(test_group_count, len(unique_groups) - 1))
    test_groups = unique_groups[:test_group_count]
    train_groups = unique_groups[test_group_count:]
    return train_groups, test_groups


def _cap_match_groups(
    groups: Sequence[str],
    match_counts: Dict[str, int],
    *,
    max_samples: Optional[int],
    random_state: int,
) -> List[str]:
    if max_samples is None:
        return list(groups)

    rng = random.Random(random_state)
    shuffled = list(groups)
    rng.shuffle(shuffled)

    kept: List[str] = []
    running_samples = 0
    for match_id in shuffled:
        if kept and running_samples >= max_samples:
            break
        match_count = int(match_counts.get(match_id, 0))
        if match_count <= 0:
            continue
        kept.append(match_id)
        running_samples += match_count
    return kept


def _scan_match_counts(champion: str, *, input_root: Path | str) -> Dict[str, int]:
    summary = load_saved_summary(champion=champion, input_root=input_root)
    total_files = int(summary.get("sequenceFiles", 0))
    counts: Counter[str] = Counter()
    progress_state = {"count": 0.0, "time": time.time()}

    processed = 0
    for processed, row in enumerate(iter_saved_metadata(champion=champion, input_root=input_root), start=1):
        valid_samples = int(row.get("validSamples") or 0)
        if valid_samples > 0:
            counts[str(row["matchId"])] += valid_samples
        _print_stage_progress(
            "scanning match counts",
            processed,
            total_files,
            state=progress_state,
        )
    if total_files == 0:
        _print_stage_progress("scanning match counts", 0, 1, force=True, state=progress_state)
    return dict(counts)


def _collect_split_metadata(
    champion: str,
    *,
    input_root: Path | str,
    train_match_ids: set[str],
    val_match_ids: set[str],
    test_match_ids: set[str],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
]:
    summary = load_saved_summary(champion=champion, input_root=input_root)
    total_files = int(summary.get("sequenceFiles", 0))
    train_metadata: List[Dict[str, Any]] = []
    val_metadata: List[Dict[str, Any]] = []
    test_metadata: List[Dict[str, Any]] = []
    train_row_types: Counter[str] = Counter()
    val_row_types: Counter[str] = Counter()
    test_row_types: Counter[str] = Counter()
    train_regions: Counter[str] = Counter()
    val_regions: Counter[str] = Counter()
    test_regions: Counter[str] = Counter()
    progress_state = {"count": 0.0, "time": time.time()}

    processed = 0
    for processed, row in enumerate(iter_saved_metadata(champion=champion, input_root=input_root), start=1):
        match_id = str(row["matchId"])
        valid_samples = int(row.get("validSamples") or 0)
        if valid_samples <= 0:
            _print_stage_progress("collecting split shards", processed, total_files, state=progress_state)
            continue

        row_type_counts = row.get("rowTypeCounts") or {}
        region_counts = row.get("regionCounts") or {}
        if match_id in train_match_ids:
            train_metadata.append(row)
            train_row_types.update({str(key): int(value) for key, value in row_type_counts.items()})
            train_regions.update({str(key): int(value) for key, value in region_counts.items()})
        elif match_id in val_match_ids:
            val_metadata.append(row)
            val_row_types.update({str(key): int(value) for key, value in row_type_counts.items()})
            val_regions.update({str(key): int(value) for key, value in region_counts.items()})
        elif match_id in test_match_ids:
            test_metadata.append(row)
            test_row_types.update({str(key): int(value) for key, value in row_type_counts.items()})
            test_regions.update({str(key): int(value) for key, value in region_counts.items()})

        _print_stage_progress("collecting split shards", processed, total_files, state=progress_state)

    return (
        train_metadata,
        val_metadata,
        test_metadata,
        dict(train_row_types),
        dict(val_row_types),
        dict(test_row_types),
        dict(train_regions),
        dict(val_regions),
        dict(test_regions),
    )


def _fit_numeric_normalizer(
    champion: str,
    *,
    input_root: Path | str,
    shard_metadata: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    feature_schema = load_saved_feature_schema(champion=champion, input_root=input_root)
    feature_count = len(feature_schema.get("numericFeatureNames") or [])
    sums = np.zeros(feature_count, dtype=np.float64)
    sum_squares = np.zeros(feature_count, dtype=np.float64)
    total_rows = 0
    progress_state = {"count": 0.0, "time": time.time()}
    combined_dataset_path = saved_dataset_paths(champion=champion, input_root=input_root)["dataset"]
    combined_numeric: Optional[np.ndarray] = None
    if combined_dataset_path.exists():
        with np.load(combined_dataset_path) as combined_dataset:
            combined_numeric = np.asarray(combined_dataset["numeric"], dtype=np.float32)

    sampled_metadata = list(shard_metadata)
    total_available_files = len(sampled_metadata)
    if MAX_NORMALIZER_FILES is not None and total_available_files > MAX_NORMALIZER_FILES:
        rng = random.Random(RANDOM_STATE + 99)
        sampled_metadata = rng.sample(sampled_metadata, MAX_NORMALIZER_FILES)
        sampled_metadata.sort(key=lambda row: str(row.get("shardPath") or ""))
        print(
            f"[train] fitting normalizer on sampled shards: {len(sampled_metadata)}/{total_available_files}",
            flush=True,
        )

    processed = 0
    total_files = len(sampled_metadata)
    for processed, row in enumerate(sampled_metadata, start=1):
        if combined_numeric is not None and row.get("rowStart") is not None and row.get("rowEnd") is not None:
            row_start = int(row.get("rowStart") or 0)
            row_end = int(row.get("rowEnd") or row_start)
            numeric = np.asarray(combined_numeric[row_start:row_end], dtype=np.float32)
        else:
            shard_path = resolve_saved_shard_path(
                str(row["shardPath"]),
                champion=champion,
                input_root=input_root,
            )
            with np.load(shard_path) as shard:
                numeric = np.asarray(shard["numeric"], dtype=np.float32)
        if numeric.size == 0:
            _print_stage_progress("fitting normalizer", processed, total_files, state=progress_state)
            continue
        sums += np.sum(numeric, axis=0)
        sum_squares += np.sum(np.square(numeric, dtype=np.float64), axis=0)
        total_rows += int(numeric.shape[0])
        _print_stage_progress("fitting normalizer", processed, total_files, state=progress_state)

    if total_rows <= 0:
        raise RuntimeError("Sequence normalizer saw zero rows.")

    mean = sums / total_rows
    variance = np.maximum((sum_squares / total_rows) - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


class SequenceShardIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        champion: str,
        input_root: Path | str,
        shard_metadata: Sequence[Dict[str, Any]],
        window_size: int,
        numeric_mean: np.ndarray,
        numeric_std: np.ndarray,
        shuffle: bool,
        random_state: int,
    ) -> None:
        super().__init__()
        self.champion = champion
        self.input_root = input_root
        self.shard_metadata = list(shard_metadata)
        self.window_size = int(window_size)
        self.numeric_mean = np.asarray(numeric_mean, dtype=np.float32)
        self.numeric_std = np.asarray(numeric_std, dtype=np.float32)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):  # type: ignore[override]
        _require_torch()
        worker = get_worker_info() if callable(get_worker_info) else None
        metadata = list(self.shard_metadata)
        rng = random.Random(self.random_state + self._epoch)
        if self.shuffle:
            rng.shuffle(metadata)

        if worker is not None:
            metadata = metadata[worker.id :: worker.num_workers]

        combined_dataset_path = saved_dataset_paths(champion=self.champion, input_root=self.input_root)["dataset"]
        combined_numeric: Optional[np.ndarray] = None
        combined_row_type: Optional[np.ndarray] = None
        combined_lane_type: Optional[np.ndarray] = None
        combined_target_region: Optional[np.ndarray] = None
        if combined_dataset_path.exists():
            with np.load(combined_dataset_path) as combined_dataset:
                combined_numeric = np.asarray(combined_dataset["numeric"], dtype=np.float32)
                combined_row_type = np.asarray(combined_dataset["rowType"], dtype=np.int64)
                combined_lane_type = np.asarray(combined_dataset["laneType"], dtype=np.int64)
                combined_target_region = np.asarray(combined_dataset["targetRegion"], dtype=np.int64)

        for meta in metadata:
            if (
                combined_numeric is not None
                and combined_row_type is not None
                and combined_lane_type is not None
                and combined_target_region is not None
                and meta.get("rowStart") is not None
                and meta.get("rowEnd") is not None
            ):
                row_start = int(meta.get("rowStart") or 0)
                row_end = int(meta.get("rowEnd") or row_start)
                numeric = np.asarray(combined_numeric[row_start:row_end], dtype=np.float32)
                row_type = np.asarray(combined_row_type[row_start:row_end], dtype=np.int64)
                lane_type = np.asarray(combined_lane_type[row_start:row_end], dtype=np.int64)
                target_region = np.asarray(combined_target_region[row_start:row_end], dtype=np.int64)
                valid_indices = np.asarray(meta.get("validSampleIndices") or [], dtype=np.int64)
                side_index = int(meta.get("sideIndex") or 0)
                enemy_champion_category_index = int(meta.get("enemyChampionCategoryIndex") or 0)
            else:
                shard_path = resolve_saved_shard_path(
                    str(meta["shardPath"]),
                    champion=self.champion,
                    input_root=self.input_root,
                )
                with np.load(shard_path) as shard:
                    numeric = np.asarray(shard["numeric"], dtype=np.float32)
                    row_type = np.asarray(shard["rowType"], dtype=np.int64)
                    lane_type = np.asarray(shard["laneType"], dtype=np.int64)
                    target_region = np.asarray(shard["targetRegion"], dtype=np.int64)
                    valid_indices = np.asarray(shard["validSampleIndices"], dtype=np.int64)
                    side_index = int(np.asarray(shard["sideIndex"], dtype=np.int64)[0])
                    enemy_champion_category_index = int(np.asarray(shard["enemyChampionCategoryIndex"], dtype=np.int64)[0])

            if valid_indices.size == 0:
                continue

            numeric = (numeric - self.numeric_mean) / self.numeric_std
            sample_indices = valid_indices.tolist()
            if self.shuffle:
                rng.shuffle(sample_indices)

            for end_idx in sample_indices:
                start_idx = max(0, int(end_idx) - self.window_size + 1)
                end_idx_int = int(end_idx)
                numeric_slice = numeric[start_idx : end_idx_int + 1]
                row_type_slice = row_type[start_idx : end_idx_int + 1]
                lane_type_slice = lane_type[start_idx : end_idx_int + 1]

                length = int(numeric_slice.shape[0])
                numeric_window = np.zeros((self.window_size, numeric.shape[1]), dtype=np.float32)
                row_type_window = np.zeros(self.window_size, dtype=np.int64)
                lane_type_window = np.zeros(self.window_size, dtype=np.int64)
                mask_window = np.zeros(self.window_size, dtype=np.float32)

                numeric_window[:length] = numeric_slice
                row_type_window[:length] = row_type_slice
                lane_type_window[:length] = lane_type_slice
                mask_window[:length] = 1.0

                yield {
                    "numeric": torch.from_numpy(numeric_window),
                    "rowType": torch.from_numpy(row_type_window),
                    "laneType": torch.from_numpy(lane_type_window),
                    "mask": torch.from_numpy(mask_window),
                    "sideIndex": torch.tensor(side_index, dtype=torch.long),
                    "enemyChampionCategoryIndex": torch.tensor(enemy_champion_category_index, dtype=torch.long),
                    "target": torch.tensor(int(target_region[end_idx_int]), dtype=torch.long),
                }


if nn is not None:
    class GRURegionClassifier(nn.Module):
        def __init__(
            self,
            *,
            numeric_dim: int,
            row_type_vocab_size: int,
            lane_type_vocab_size: int,
            enemy_champion_category_vocab_size: int,
            hidden_size: int,
            embed_dim: int,
            gru_layers: int,
            dropout_p: float,
            num_classes: int,
        ) -> None:
            super().__init__()
            self.row_type_embedding = nn.Embedding(row_type_vocab_size, embed_dim, padding_idx=0)
            self.lane_type_embedding = nn.Embedding(lane_type_vocab_size, embed_dim, padding_idx=0)
            self.side_embedding = nn.Embedding(2, 2)
            self.enemy_champion_category_embedding = nn.Embedding(enemy_champion_category_vocab_size, embed_dim, padding_idx=0)

            input_dim = numeric_dim + (embed_dim * 3) + 2
            self.encoder = nn.GRU(
                input_dim,
                hidden_size,
                num_layers=gru_layers,
                dropout=dropout_p if gru_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(p=dropout_p)
            self.head = nn.Linear(hidden_size, num_classes)

        def forward(self, batch: Dict[str, Any]) -> Any:
            numeric = batch["numeric"].float()
            row_type_embed = self.row_type_embedding(batch["rowType"].long())
            lane_type_embed = self.lane_type_embedding(batch["laneType"].long())

            batch_size, seq_len, _ = numeric.shape
            side_embed = self.side_embedding(batch["sideIndex"].long()).unsqueeze(1).expand(batch_size, seq_len, -1)
            enemy_champion_category_embed = self.enemy_champion_category_embedding(batch["enemyChampionCategoryIndex"].long()).unsqueeze(1).expand(batch_size, seq_len, -1)

            features = torch.cat(
                [
                    numeric,
                    row_type_embed,
                    lane_type_embed,
                    side_embed,
                    enemy_champion_category_embed,
                ],
                dim=-1,
            )
            lengths = torch.clamp(batch["mask"].sum(dim=1).long(), min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                features,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.encoder(packed)
            last_hidden = hidden[-1]
            return self.head(self.dropout(last_hidden))
else:
    class GRURegionClassifier:  # pragma: no cover - import fallback only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def _move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _clone_state_dict_to_cpu(model: Any) -> Dict[str, Any]:
    _require_torch()
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in model.state_dict().items()
    }


def _build_class_weights(region_counts: Dict[str, int], *, device: str) -> Any:
    _require_torch()
    counts = np.asarray(
        [max(1, int(region_counts.get(region_name, 0))) for region_name in REGION_NAMES],
        dtype=np.float64,
    )
    weights = np.sqrt(np.sum(counts) / counts)
    weights /= max(float(np.mean(weights)), 1e-12)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_epoch(
    model: GRURegionClassifier,
    loader: Any,
    *,
    optimizer: Any,
    criterion: Any,
    device: str,
    total_samples: int,
    epoch_index: int,
) -> Dict[str, float]:
    _require_torch()
    model.train()
    processed = 0
    running_loss = 0.0
    progress_state = {"count": 0.0, "time": time.time()}
    print(f"[train] starting epoch {epoch_index + 1}/{EPOCHS}", flush=True)

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch["target"])
        loss.backward()
        if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        batch_size = int(batch["target"].shape[0])
        processed += batch_size
        running_loss += float(loss.detach().item()) * batch_size
        _print_stage_progress(
            f"training epoch {epoch_index + 1}/{EPOCHS}",
            processed,
            total_samples,
            count_every=TRAIN_PROGRESS_PRINT_EVERY_COUNT,
            seconds_every=TRAIN_PROGRESS_PRINT_EVERY_SECONDS,
            state=progress_state,
        )

    return {
        "loss": (running_loss / processed) if processed else 0.0,
    }


def _evaluate_model(
    model: GRURegionClassifier,
    loader: Any,
    *,
    device: str,
    total_samples: int,
    label: str = "evaluating",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    _require_torch()
    model.eval()
    correct = 0
    top2_correct = 0
    top3_correct = 0
    neg_log_likelihood = 0.0
    example_probabilities: Dict[str, float] = {}
    per_class_correct: Counter[str] = Counter()
    per_class_total: Counter[str] = Counter()
    processed = 0
    progress_state = {"count": 0.0, "time": time.time()}

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            targets = batch["target"]

            correct += int((predictions == targets).sum().item())
            top3 = torch.topk(probabilities, k=3, dim=1).indices
            top2 = top3[:, :2]
            top3_correct += int((top3 == targets.unsqueeze(1)).any(dim=1).sum().item())
            top2_correct += int((top2 == targets.unsqueeze(1)).any(dim=1).sum().item())
            target_cpu = targets.detach().cpu().tolist()
            prediction_cpu = predictions.detach().cpu().tolist()
            for target_idx, predicted_idx in zip(target_cpu, prediction_cpu):
                region_name = REGION_NAMES[int(target_idx)]
                per_class_total[region_name] += 1
                if int(target_idx) == int(predicted_idx):
                    per_class_correct[region_name] += 1

            true_probs = torch.clamp(
                probabilities[torch.arange(targets.shape[0], device=targets.device), targets],
                min=1e-15,
                max=1.0,
            )
            neg_log_likelihood += float((-torch.log(true_probs)).sum().item())

            if not example_probabilities and probabilities.shape[0] > 0:
                first_row = probabilities[0].detach().cpu().tolist()
                example_probabilities = {
                    region_name: float(first_row[idx]) for idx, region_name in enumerate(REGION_NAMES)
                }

            processed += int(targets.shape[0])
            _print_stage_progress(label, processed, total_samples, state=progress_state)

    per_class_accuracy = {
        region_name: (
            float(per_class_correct.get(region_name, 0)) / float(per_class_total.get(region_name, 0))
            if per_class_total.get(region_name, 0)
            else 0.0
        )
        for region_name in REGION_NAMES
    }
    active_class_accuracies = [
        per_class_accuracy[region_name]
        for region_name in REGION_NAMES
        if per_class_total.get(region_name, 0)
    ]
    metrics = {
        "accuracy": (correct / total_samples) if total_samples else 0.0,
        "top2Accuracy": (top2_correct / total_samples) if total_samples else 0.0,
        "top3Accuracy": (top3_correct / total_samples) if total_samples else 0.0,
        "logLoss": (neg_log_likelihood / total_samples) if total_samples else 0.0,
        "macroAccuracy": (
            float(sum(active_class_accuracies)) / float(len(active_class_accuracies))
            if active_class_accuracies
            else 0.0
        ),
        "perClassAccuracy": per_class_accuracy,
    }
    return metrics, example_probabilities


def _majority_baseline(
    train_region_counts: Dict[str, int],
    test_region_counts: Dict[str, int],
    *,
    total_test_samples: int,
) -> Dict[str, float]:
    majority_region = max(train_region_counts.items(), key=lambda item: item[1])[0]
    majority_hits = int(test_region_counts.get(majority_region, 0))
    return {
        "majorityClassIndex": REGION_NAMES.index(majority_region),
        "majorityClassName": majority_region,
        "accuracy": (majority_hits / total_test_samples) if total_test_samples else 0.0,
    }


def _print_training_overview(
    *,
    summary: Dict[str, Any],
    train_match_ids: Sequence[str],
    val_match_ids: Sequence[str],
    test_match_ids: Sequence[str],
    train_samples: int,
    val_samples: int,
    test_samples: int,
    device: str,
) -> None:
    total_matches = len(train_match_ids) + len(val_match_ids) + len(test_match_ids)
    train_ratio = (len(train_match_ids) / total_matches) if total_matches else 0.0
    val_ratio = (len(val_match_ids) / total_matches) if total_matches else 0.0
    test_ratio = (len(test_match_ids) / total_matches) if total_matches else 0.0

    print("Training overview")
    print(f"- Champion: {MODEL_CHAMPION}")
    print(f"- Input dir: {champion_input_dir(MODEL_CHAMPION, input_root=INPUT_DATA_ROOT)}")
    print(f"- Sequence files used to build input: {summary.get('sequenceFiles', 0)}")
    print(f"- Unique matches loaded: {total_matches}")
    print(f"- Train/val/test split is by matchId, not by individual windows")
    print(
        f"- Match split: train={len(train_match_ids)} ({train_ratio:.1%}), "
        f"val={len(val_match_ids)} ({val_ratio:.1%}), "
        f"test={len(test_match_ids)} ({test_ratio:.1%})"
    )
    print(
        f"- Sample split: train={train_samples}, val={val_samples}, test={test_samples}, "
        f"total={int(summary.get('samples', 0))}"
    )
    print(
        f"- Sample caps: train={MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else 'ALL'}, "
        f"val={MAX_VAL_SAMPLES if MAX_VAL_SAMPLES is not None else 'ALL'}, "
        f"test={MAX_TEST_SAMPLES if MAX_TEST_SAMPLES is not None else 'ALL'}"
    )
    print(f"- Window size: {WINDOW_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Hidden size: {HIDDEN_SIZE}")
    print(f"- Embedding dim: {EMBED_DIM}")
    print(f"- Device: {device}")
    print(f"- Random seed: {RANDOM_STATE}")
    print(flush=True)


def _build_summary(
    *,
    dataset_summary: Dict[str, Any],
    feature_schema: Dict[str, Any],
    train_match_ids: Sequence[str],
    val_match_ids: Sequence[str],
    test_match_ids: Sequence[str],
    train_samples: int,
    val_samples: int,
    test_samples: int,
    train_row_types: Dict[str, int],
    val_row_types: Dict[str, int],
    test_row_types: Dict[str, int],
    train_regions: Dict[str, int],
    val_regions: Dict[str, int],
    test_regions: Dict[str, int],
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    example_probabilities: Dict[str, float],
    class_weights: Sequence[float],
    training_history: Sequence[Dict[str, Any]],
    best_epoch: int,
    device: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_summary,
        "trainSamples": train_samples,
        "valSamples": val_samples,
        "testSamples": test_samples,
        "trainMatches": len(train_match_ids),
        "valMatches": len(val_match_ids),
        "testMatches": len(test_match_ids),
        "trainRowTypes": train_row_types,
        "valRowTypes": val_row_types,
        "testRowTypes": test_row_types,
        "trainRegions": train_regions,
        "valRegions": val_regions,
        "testRegions": test_regions,
        "windowSize": WINDOW_SIZE,
        "numericFeatureNames": list(feature_schema.get("numericFeatureNames") or []),
        "targetNames": list(feature_schema.get("targetNames") or []),
        "model": {
            "type": "GRURegionClassifier",
            "hiddenSize": HIDDEN_SIZE,
            "embedDim": EMBED_DIM,
            "gruLayers": GRU_LAYERS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batchSize": BATCH_SIZE,
            "learningRate": LEARNING_RATE,
            "weightDecay": WEIGHT_DECAY,
            "random_state": RANDOM_STATE,
            "device": device,
        },
        "training": {
            "maxGradNorm": MAX_GRAD_NORM,
            "earlyStoppingPatience": EARLY_STOPPING_PATIENCE,
            "lrSchedulerPatience": LR_SCHEDULER_PATIENCE,
            "lrSchedulerFactor": LR_SCHEDULER_FACTOR,
            "maxNormalizerFiles": MAX_NORMALIZER_FILES,
            "classWeights": {region_name: float(class_weights[idx]) for idx, region_name in enumerate(REGION_NAMES)},
            "history": list(training_history),
            "bestEpoch": best_epoch,
        },
        "validationMetrics": validation_metrics,
        "metrics": test_metrics,
        "majorityBaseline": _majority_baseline(
            train_regions,
            test_regions,
            total_test_samples=test_samples,
        ),
        "exampleProbabilities": example_probabilities,
    }


def main() -> None:
    _require_torch()
    device = _default_device()
    dataset_summary = load_saved_summary(champion=MODEL_CHAMPION, input_root=INPUT_DATA_ROOT)
    total_samples = int(dataset_summary.get("samples", 0))
    if total_samples <= 0:
        raise RuntimeError("Saved dataset reports zero samples.")

    match_counts = _scan_match_counts(MODEL_CHAMPION, input_root=INPUT_DATA_ROOT)
    all_match_ids = sorted(match_counts)
    train_pool_match_ids, test_match_ids = _group_train_test_match_ids(
        all_match_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_match_ids, val_match_ids = _group_train_test_match_ids(
        train_pool_match_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE + 17,
    )
    train_match_ids = _cap_match_groups(
        train_match_ids,
        match_counts,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )
    val_match_ids = _cap_match_groups(
        val_match_ids,
        match_counts,
        max_samples=MAX_VAL_SAMPLES,
        random_state=RANDOM_STATE + 2,
    )
    test_match_ids = _cap_match_groups(
        test_match_ids,
        match_counts,
        max_samples=MAX_TEST_SAMPLES,
        random_state=RANDOM_STATE + 1,
    )
    train_match_id_set = set(train_match_ids)
    val_match_id_set = set(val_match_ids)
    test_match_id_set = set(test_match_ids)

    (
        train_metadata,
        val_metadata,
        test_metadata,
        train_row_types,
        val_row_types,
        test_row_types,
        train_regions,
        val_regions,
        test_regions,
    ) = _collect_split_metadata(
        MODEL_CHAMPION,
        input_root=INPUT_DATA_ROOT,
        train_match_ids=train_match_id_set,
        val_match_ids=val_match_id_set,
        test_match_ids=test_match_id_set,
    )
    train_samples = sum(int(row.get("validSamples") or 0) for row in train_metadata)
    val_samples = sum(int(row.get("validSamples") or 0) for row in val_metadata)
    test_samples = sum(int(row.get("validSamples") or 0) for row in test_metadata)
    if train_samples == 0 or val_samples == 0 or test_samples == 0:
        raise RuntimeError("Train/val/test split produced zero selected sequence samples.")

    _print_training_overview(
        summary=dataset_summary,
        train_match_ids=train_match_ids,
        val_match_ids=val_match_ids,
        test_match_ids=test_match_ids,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        device=device,
    )

    feature_schema = load_saved_feature_schema(champion=MODEL_CHAMPION, input_root=INPUT_DATA_ROOT)
    numeric_mean, numeric_std = _fit_numeric_normalizer(
        MODEL_CHAMPION,
        input_root=INPUT_DATA_ROOT,
        shard_metadata=train_metadata,
    )

    train_dataset = SequenceShardIterableDataset(
        champion=MODEL_CHAMPION,
        input_root=INPUT_DATA_ROOT,
        shard_metadata=train_metadata,
        window_size=WINDOW_SIZE,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    test_dataset = SequenceShardIterableDataset(
        champion=MODEL_CHAMPION,
        input_root=INPUT_DATA_ROOT,
        shard_metadata=test_metadata,
        window_size=WINDOW_SIZE,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        shuffle=False,
        random_state=RANDOM_STATE,
    )
    val_dataset = SequenceShardIterableDataset(
        champion=MODEL_CHAMPION,
        input_root=INPUT_DATA_ROOT,
        shard_metadata=val_metadata,
        window_size=WINDOW_SIZE,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        shuffle=False,
        random_state=RANDOM_STATE,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    row_type_vocab_size = int((feature_schema.get("categoricalFeatures") or {}).get("rowType", {}).get("unknownIndex", 0)) + 1
    lane_type_vocab_size = int((feature_schema.get("categoricalFeatures") or {}).get("laneType", {}).get("unknownIndex", 0)) + 1
    enemy_champion_category_vocab_size = int((feature_schema.get("categoricalFeatures") or {}).get("enemyChampionCategory", {}).get("unknownIndex", 0)) + 1
    numeric_dim = len(feature_schema.get("numericFeatureNames") or [])

    model = GRURegionClassifier(
        numeric_dim=numeric_dim,
        row_type_vocab_size=row_type_vocab_size,
        lane_type_vocab_size=lane_type_vocab_size,
        enemy_champion_category_vocab_size=enemy_champion_category_vocab_size,
        hidden_size=HIDDEN_SIZE,
        embed_dim=EMBED_DIM,
        gru_layers=GRU_LAYERS,
        dropout_p=DROPOUT,
        num_classes=len(REGION_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    class_weights = _build_class_weights(train_regions, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    best_state_dict: Optional[Dict[str, Any]] = None
    best_epoch = 0
    best_val_accuracy = -1.0
    best_val_log_loss = float("inf")
    epochs_without_improvement = 0
    training_history: List[Dict[str, Any]] = []
    best_validation_metrics: Dict[str, float] = {}

    for epoch in range(EPOCHS):
        train_dataset.set_epoch(epoch)
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            total_samples=train_samples,
            epoch_index=epoch,
        )
        val_metrics, _ = _evaluate_model(
            model,
            val_loader,
            device=device,
            total_samples=val_samples,
            label=f"validating epoch {epoch + 1}/{EPOCHS}",
        )
        scheduler.step(float(val_metrics["logLoss"]))
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_row = {
            "epoch": epoch + 1,
            "trainLoss": float(train_metrics["loss"]),
            "valAccuracy": float(val_metrics["accuracy"]),
            "valTop2Accuracy": float(val_metrics["top2Accuracy"]),
            "valTop3Accuracy": float(val_metrics["top3Accuracy"]),
            "valLogLoss": float(val_metrics["logLoss"]),
            "valMacroAccuracy": float(val_metrics["macroAccuracy"]),
            "learningRate": current_lr,
        }
        training_history.append(history_row)
        print(f"[train] epoch summary: {json.dumps(history_row, separators=(',', ':'))}", flush=True)

        improved = (
            float(val_metrics["accuracy"]) > best_val_accuracy
            or (
                float(val_metrics["accuracy"]) == best_val_accuracy
                and float(val_metrics["logLoss"]) < best_val_log_loss
            )
        )
        if improved:
            best_epoch = epoch + 1
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_log_loss = float(val_metrics["logLoss"])
            best_validation_metrics = dict(val_metrics)
            best_state_dict = _clone_state_dict_to_cpu(model)
            epochs_without_improvement = 0
            print(f"[train] new best checkpoint at epoch {best_epoch}", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"[train] no validation improvement for {epochs_without_improvement} epoch(s)",
                flush=True,
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("[train] early stopping triggered", flush=True)
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics, example_probabilities = _evaluate_model(
        model,
        test_loader,
        device=device,
        total_samples=test_samples,
        label="testing best checkpoint",
    )
    final_summary = _build_summary(
        dataset_summary=dataset_summary,
        feature_schema=feature_schema,
        train_match_ids=train_match_ids,
        val_match_ids=val_match_ids,
        test_match_ids=test_match_ids,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_row_types=train_row_types,
        val_row_types=val_row_types,
        test_row_types=test_row_types,
        train_regions=train_regions,
        val_regions=val_regions,
        test_regions=test_regions,
        validation_metrics=best_validation_metrics,
        test_metrics=metrics,
        example_probabilities=example_probabilities,
        class_weights=class_weights.detach().cpu().tolist(),
        training_history=training_history,
        best_epoch=best_epoch,
        device=device,
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization": {
                "mean": numeric_mean.tolist(),
                "std": numeric_std.tolist(),
            },
            "feature_schema": feature_schema,
            "summary": final_summary,
            "config": {
                "windowSize": WINDOW_SIZE,
                "hiddenSize": HIDDEN_SIZE,
                "embedDim": EMBED_DIM,
                "gruLayers": GRU_LAYERS,
                "dropout": DROPOUT,
            },
        },
        MODEL_OUT,
    )
    SUMMARY_OUT.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print(json.dumps(final_summary, indent=2))
    print(f"Saved model bundle to {MODEL_OUT}")
    print(f"Saved summary to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()

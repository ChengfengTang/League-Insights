"""
Turn processed `Data/log/` files into sequence-ready training input.

The log format stores one JSON file per jungler:
  Data/log/<run>/<ChampionName>/<matchId>_p<participantId>_<side>.json

This module exports fixed-window sequence shards under `Train/Input/` for the
default GRU training pipeline in `Train/train.py`.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]

try:
    from champCategory import CHAMPION_CATEGORIES, get_champion_category
except ModuleNotFoundError:
    from Train.champCategory import CHAMPION_CATEGORIES, get_champion_category

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = REPO_ROOT / "Data" / "log"
LOG_ROOT = DEFAULT_LOG_ROOT
CHAMPION = "LeeSin"  # Change this to load one champion across all runs.
INPUT_ROOT = REPO_ROOT / "Train" / "Input"
MAP_MAX_COORD = 15000.0
MAP_CENTER = MAP_MAX_COORD / 2.0
LANING_PHASE_MS = 14 * 60 * 1000
PROGRESS_BAR_WIDTH = 30
PROGRESS_PRINT_EVERY_COUNT = 100
PROGRESS_PRINT_EVERY_SECONDS = 10.0
WINDOW_SIZE = 16
MAX_FILES: Optional[int] = None
BASE_COORDS_BY_TEAM = {
    100: (1500.0, 1500.0),
    200: (13500.0, 13500.0),
}
ROW_TYPES = (
    "frame",
    "kill",
    "assist",
    "death",
    "respawn",
    "dragon",
    "plate",
    "backing",
)
CATEGORY_NAMES = tuple(CHAMPION_CATEGORIES.keys())
LANE_TYPES = ("TOP_LANE", "MID_LANE", "BOT_LANE", "UNKNOWN")
REGION_NAMES = (
    "top_lane",
    "mid_lane",
    "bot_lane",
    "own_top_jg",
    "own_bot_jg",
    "enemy_top_jg",
    "enemy_bot_jg",
)
REGION_TO_INDEX = {name: idx for idx, name in enumerate(REGION_NAMES)}
REGION_CENTERS = {
    "top_lane": (2500.0, 11500.0),
    "mid_lane": (7500.0, 7500.0),
    "bot_lane": (11500.0, 2500.0),
    "red_top_jg": (7000.0, 11000.0),
    "red_bot_jg": (11000.0, 7000.0),
    "blue_top_jg": (4000.0, 8000.0),
    "blue_bot_jg": (8000.0, 4000.0),
    "top_river": (6000.0, 9000.0),
    "bot_river": (9000.0, 6000.0),
}
ROW_TYPE_TO_INDEX = {str(name).upper(): idx + 1 for idx, name in enumerate(ROW_TYPES)}
ROW_TYPE_UNKNOWN_INDEX = len(ROW_TYPES) + 1
LANE_TYPE_TO_INDEX = {str(name).upper(): idx + 1 for idx, name in enumerate(LANE_TYPES)}
CATEGORY_TO_INDEX = {str(name).upper(): idx + 1 for idx, name in enumerate(CATEGORY_NAMES)}
CATEGORY_UNKNOWN_INDEX = len(CATEGORY_NAMES) + 1

NUMERIC_FEATURE_NAMES = [
    "x",
    "y",
    "timestampMinutes",
    "phase01",
    "level",
    "csJungle",
    "csLane",
    "currentGold",
    "totalGold",
    "movementSpeed",
    "frameIndex",
    "distToOwnBase",
    "distToEnemyBase",
    "distToCenter",
    "distToOwnBaseNorm",
    "distToEnemyBaseNorm",
    "distToCenterNorm",
    "positionKnown",
    "positionImputed",
    "prevDtSeconds",
    "prevDx",
    "prevDy",
    "prevDistance",
    "prevSpeed",
]


def format_progress_bar(done: int, total: int) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int((done / total) * PROGRESS_BAR_WIDTH)
    bar = "=" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    return f"[{bar}] {done}/{total}"


def _safe_name(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum())
    return cleaned or "Unknown"


def _iter_log_paths(
    log_root: Path,
    champion: Optional[str] = None,
    limit_files: Optional[int] = None,
) -> List[Path]:
    root = log_root.resolve()
    if root.is_file():
        return [root]
    if champion:
        champion_key = "".join(ch.lower() for ch in str(champion) if ch.isalnum())
        champion_dirs: List[Path] = []

        def maybe_add_dir(path: Path) -> None:
            if not path.is_dir():
                return
            path_key = "".join(ch.lower() for ch in path.name if ch.isalnum())
            if path_key == champion_key:
                champion_dirs.append(path)

        maybe_add_dir(root)
        for child in root.iterdir():
            maybe_add_dir(child)
            if child.is_dir():
                for grandchild in child.iterdir():
                    maybe_add_dir(grandchild)

        paths = sorted(
            path
            for champion_dir in champion_dirs
            for path in champion_dir.glob("*.json")
            if path.is_file()
        )
        if limit_files is not None:
            return paths[: max(0, limit_files)]
        return paths

    if limit_files is not None:
        paths: List[Path] = []
        for path in root.rglob("*.json"):
            if not path.is_file():
                continue
            paths.append(path)
            if len(paths) >= max(0, limit_files):
                break
        return sorted(paths)

    return sorted(path for path in root.rglob("*.json") if path.is_file())


def _float_or(default: float, value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _resolve_row_position(
    row: Dict[str, Any],
    team_id: int,
) -> Tuple[Optional[float], Optional[float], bool]:
    x = row.get("x")
    y = row.get("y")
    imputed = False

    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        if row.get("rowType") in {"backing", "respawn"}:
            x, y = BASE_COORDS_BY_TEAM.get(team_id, (None, None))
            imputed = True

    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None, None, imputed

    return float(x), float(y), imputed


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _normalize_lane_type(value: Any) -> str:
    lane = str(value or "UNKNOWN").strip().upper()
    return lane if lane in LANE_TYPES else "UNKNOWN"


def _classify_region(x: float, y: float) -> str:
    return min(
        REGION_CENTERS,
        key=lambda region_name: _distance(
            x,
            y,
            REGION_CENTERS[region_name][0],
            REGION_CENTERS[region_name][1],
        ),
    )


def _indexed_value(mapping: Dict[str, int], value: Any, *, unknown_index: int) -> int:
    key = str(value or "").strip().upper()
    return int(mapping.get(key, unknown_index))


def _canonicalize_target_region(raw_region_name: str, *, is_blue_side: bool) -> Optional[str]:
    if raw_region_name in {"top_lane", "mid_lane", "bot_lane"}:
        return raw_region_name
    if raw_region_name in {"top_river", "bot_river"}:
        return None

    if is_blue_side:
        region_map = {
            "blue_top_jg": "own_top_jg",
            "blue_bot_jg": "own_bot_jg",
            "red_top_jg": "enemy_top_jg",
            "red_bot_jg": "enemy_bot_jg",
        }
    else:
        region_map = {
            "red_top_jg": "own_top_jg",
            "red_bot_jg": "own_bot_jg",
            "blue_top_jg": "enemy_top_jg",
            "blue_bot_jg": "enemy_bot_jg",
        }
    return region_map.get(raw_region_name)


def _resolve_target_region(
    rows: Sequence[Dict[str, Any]],
    resolved_positions: Sequence[Tuple[Optional[float], Optional[float], bool]],
    *,
    current_idx: int,
    is_blue_side: bool,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    current_timestamp_ms = int(rows[current_idx].get("timestampMs") or 0)

    for next_idx in range(current_idx + 1, len(rows)):
        next_row = rows[next_idx]
        next_row_type = str(next_row.get("rowType") or "frame")
        if next_row_type in {"backing", "respawn"}:
            return None, None, None, "base_target_row"

        next_x, next_y, _ = resolved_positions[next_idx]
        if next_x is None or next_y is None:
            return None, None, None, "missing_target_position"

        next_timestamp_ms = int(next_row.get("timestampMs") or 0)
        horizon_ms = next_timestamp_ms - current_timestamp_ms
        if horizon_ms <= 0:
            return None, None, None, "non_positive_horizon"

        raw_region_name = _classify_region(float(next_x), float(next_y))
        canonical_region_name = _canonicalize_target_region(raw_region_name, is_blue_side=is_blue_side)
        if canonical_region_name is None:
            continue

        return canonical_region_name, next_idx, horizon_ms, None

    return None, None, None, "river_only_future"


def champion_input_dir(champion: str, input_root: Path | str = INPUT_ROOT) -> Path:
    return Path(input_root).resolve() / _safe_name(champion)


def saved_dataset_paths(
    champion: str,
    *,
    input_root: Path | str = INPUT_ROOT,
) -> Dict[str, Path]:
    input_dir = champion_input_dir(champion, input_root=input_root)
    return {
        "input_dir": input_dir,
        "summary": input_dir / "summary.json",
        "feature_schema": input_dir / "feature_schema.json",
        "sample_metadata": input_dir / "sequence_metadata.jsonl",
        "dataset": input_dir / "dataset.npz",
    }


def load_saved_summary(
    *,
    champion: str,
    input_root: Path | str = INPUT_ROOT,
) -> Dict[str, Any]:
    paths = saved_dataset_paths(champion, input_root=input_root)
    if not paths["summary"].exists():
        raise RuntimeError(f"Saved summary not found: {paths['summary']}")
    return json.loads(paths["summary"].read_text(encoding="utf-8"))


def load_saved_feature_schema(
    *,
    champion: str,
    input_root: Path | str = INPUT_ROOT,
) -> Dict[str, Any]:
    paths = saved_dataset_paths(champion, input_root=input_root)
    if not paths["feature_schema"].exists():
        raise RuntimeError(f"Saved feature schema not found: {paths['feature_schema']}")
    return json.loads(paths["feature_schema"].read_text(encoding="utf-8"))


def iter_saved_metadata(
    *,
    champion: str,
    input_root: Path | str = INPUT_ROOT,
) -> Iterator[Dict[str, Any]]:
    paths = saved_dataset_paths(champion, input_root=input_root)
    if not paths["sample_metadata"].exists():
        raise RuntimeError(f"Saved sample metadata not found: {paths['sample_metadata']}")

    with paths["sample_metadata"].open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_saved_shard_path(
    shard_relative_path: str,
    *,
    champion: str,
    input_root: Path | str = INPUT_ROOT,
) -> Path:
    paths = saved_dataset_paths(champion, input_root=input_root)
    return paths["input_dir"] / shard_relative_path


def _build_step_numeric_features(
    *,
    payload: Dict[str, Any],
    row: Dict[str, Any],
    resolved_position: Tuple[Optional[float], Optional[float], bool],
    prev_position: Tuple[Optional[float], Optional[float], bool],
) -> List[float]:
    team_id = int(payload.get("teamId") or 100)
    own_base_x, own_base_y = BASE_COORDS_BY_TEAM.get(team_id, BASE_COORDS_BY_TEAM[100])
    enemy_team_id = 200 if team_id == 100 else 100
    enemy_base_x, enemy_base_y = BASE_COORDS_BY_TEAM.get(enemy_team_id, BASE_COORDS_BY_TEAM[200])

    row_x_raw, row_y_raw, row_position_imputed = resolved_position
    prev_x_raw, prev_y_raw, _ = prev_position
    position_known = row_x_raw is not None and row_y_raw is not None
    row_x = float(row_x_raw) if row_x_raw is not None else 0.0
    row_y = float(row_y_raw) if row_y_raw is not None else 0.0

    prev_known = prev_x_raw is not None and prev_y_raw is not None and position_known
    prev_x = float(prev_x_raw) if prev_x_raw is not None else 0.0
    prev_y = float(prev_y_raw) if prev_y_raw is not None else 0.0

    timestamp_ms = int(row.get("timestampMs") or 0)
    minutes = timestamp_ms / 60000.0
    phase = min(1.0, timestamp_ms / LANING_PHASE_MS) if LANING_PHASE_MS > 0 else 0.0

    dist_own_base = _distance(row_x, row_y, own_base_x, own_base_y)
    dist_enemy_base = _distance(row_x, row_y, enemy_base_x, enemy_base_y)
    dist_center = _distance(row_x, row_y, MAP_CENTER, MAP_CENTER)

    prev_dt_ms = (
        timestamp_ms - int(row.get("_prevTimestampMs") or 0)
        if row.get("_prevTimestampMs") is not None
        else 0
    )
    prev_dx = (row_x - prev_x) if prev_known else 0.0
    prev_dy = (row_y - prev_y) if prev_known else 0.0
    prev_distance = math.hypot(prev_dx, prev_dy) if prev_known else 0.0
    prev_speed = (prev_distance / (prev_dt_ms / 1000.0)) if prev_known and prev_dt_ms > 0 else 0.0

    return [
        row_x,
        row_y,
        minutes,
        phase,
        _float_or(0.0, row.get("level")),
        _float_or(0.0, row.get("csJungle")),
        _float_or(0.0, row.get("csLane")),
        _float_or(0.0, row.get("currentGold")),
        _float_or(0.0, row.get("totalGold")),
        _float_or(0.0, row.get("movementSpeed")),
        _float_or(0.0, row.get("frameIndex")),
        dist_own_base,
        dist_enemy_base,
        dist_center,
        dist_own_base / MAP_MAX_COORD,
        dist_enemy_base / MAP_MAX_COORD,
        dist_center / MAP_MAX_COORD,
        1.0 if position_known else 0.0,
        1.0 if row_position_imputed else 0.0,
        prev_dt_ms / 1000.0,
        prev_dx,
        prev_dy,
        prev_distance,
        prev_speed,
    ]


def export_dataset(
    log_root: Path | str = LOG_ROOT,
    *,
    champion: Optional[str] = None,
    input_root: Path | str = INPUT_ROOT,
    window_size: int = WINDOW_SIZE,
    limit_files: Optional[int] = MAX_FILES,
    include_row_types: Optional[Iterable[str]] = None,
) -> Path:
    paths = saved_dataset_paths(champion or "ALL", input_root=input_root)
    paths["input_dir"].mkdir(parents=True, exist_ok=True)

    allowed_row_types = set(include_row_types) if include_row_types is not None else None
    source_root = Path(log_root).resolve()
    source_paths = _iter_log_paths(source_root, champion=champion, limit_files=limit_files)

    summary_row_type_counts: Counter[str] = Counter()
    summary_region_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    total_valid_samples = 0
    total_rows = 0

    processed_paths = 0
    last_progress_count = 0
    last_progress_time = time.time()

    feature_schema = {
        "windowSize": int(window_size),
        "numericFeatureNames": list(NUMERIC_FEATURE_NAMES),
        "targetNames": list(REGION_NAMES),
        "categoricalFeatures": {
            "rowType": {
                "padIndex": 0,
                "unknownIndex": ROW_TYPE_UNKNOWN_INDEX,
                "names": list(ROW_TYPES),
            },
            "laneType": {
                "padIndex": 0,
                "unknownIndex": LANE_TYPE_TO_INDEX["UNKNOWN"],
                "names": list(LANE_TYPES),
            },
            "enemyChampionCategory": {
                "padIndex": 0,
                "unknownIndex": CATEGORY_UNKNOWN_INDEX,
                "names": list(CATEGORY_NAMES),
            },
            "side": {
                "padIndex": 0,
                "names": ["red", "blue"],
            },
        },
    }

    print(
        f"[loadData] champion={champion or 'ALL'} {format_progress_bar(0, max(1, len(source_paths)))}",
        flush=True,
    )

    all_numeric_rows: List[np.ndarray] = []
    all_row_type_rows: List[np.ndarray] = []
    all_lane_type_rows: List[np.ndarray] = []
    all_target_region_rows: List[np.ndarray] = []
    global_row_start = 0

    with paths["sample_metadata"].open("w", encoding="utf-8") as metadata_handle:
        for path in source_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows") or []

            if len(rows) < 2:
                skipped_counts["files_too_small"] += 1
                processed_paths += 1
                continue

            team_id = int(payload.get("teamId") or 100)
            is_blue_side = bool(payload.get("isBlueSide"))
            enemy_champion_category_name = get_champion_category(payload.get("enemyChampionName"), primary=True)

            resolved_positions = [_resolve_row_position(row, team_id) for row in rows]
            numeric_rows: List[List[float]] = []
            row_type_indices: List[int] = []
            lane_type_indices: List[int] = []

            for idx, row in enumerate(rows):
                prev_row = rows[idx - 1] if idx > 0 else None
                row_copy = dict(row)
                row_copy["_prevTimestampMs"] = int(prev_row.get("timestampMs") or 0) if prev_row is not None else None
                numeric_rows.append(
                    _build_step_numeric_features(
                        payload=payload,
                        row=row_copy,
                        resolved_position=resolved_positions[idx],
                        prev_position=resolved_positions[idx - 1] if idx > 0 else (None, None, False),
                    )
                )
                row_type_indices.append(
                    _indexed_value(
                        ROW_TYPE_TO_INDEX,
                        row.get("rowType"),
                        unknown_index=ROW_TYPE_UNKNOWN_INDEX,
                    )
                )
                lane_type_indices.append(
                    _indexed_value(
                        LANE_TYPE_TO_INDEX,
                        _normalize_lane_type(row.get("laneType")),
                        unknown_index=LANE_TYPE_TO_INDEX["UNKNOWN"],
                    )
                )

            target_region_indices = np.full(len(rows), -1, dtype=np.int16)
            valid_sample_indices: List[int] = []
            shard_row_type_counts: Counter[str] = Counter()
            shard_region_counts: Counter[str] = Counter()

            for idx in range(len(rows) - 1):
                row = rows[idx]
                next_row = rows[idx + 1]
                row_type = str(row.get("rowType") or "frame")
                next_row_type = str(next_row.get("rowType") or "frame")
                summary_row_type_counts[row_type] += 1
                shard_row_type_counts[row_type] += 1

                if allowed_row_types is not None and row_type not in allowed_row_types:
                    skipped_counts["row_type_filtered"] += 1
                    continue

                row_x, row_y, _ = resolved_positions[idx]
                next_x, next_y, _ = resolved_positions[idx + 1]
                if row_x is None or row_y is None:
                    skipped_counts["missing_current_position"] += 1
                    continue
                target_region_name, target_idx, _, skip_reason = _resolve_target_region(
                    rows,
                    resolved_positions,
                    current_idx=idx,
                    is_blue_side=is_blue_side,
                )
                if target_region_name is None or target_idx is None:
                    skipped_counts[str(skip_reason or "missing_target_region")] += 1
                    continue

                target_region_index = REGION_TO_INDEX[target_region_name]
                target_region_indices[idx] = target_region_index
                valid_sample_indices.append(idx)
                total_valid_samples += 1
                summary_region_counts[target_region_name] += 1
                shard_region_counts[target_region_name] += 1

            total_rows += len(rows)
            numeric_array = np.asarray(numeric_rows, dtype=np.float32)
            row_type_array = np.asarray(row_type_indices, dtype=np.int16)
            lane_type_array = np.asarray(lane_type_indices, dtype=np.int16)
            target_region_array = np.asarray(target_region_indices, dtype=np.int16)
            valid_indices_array = np.asarray(valid_sample_indices, dtype=np.int32)
            side_index_value = int(1 if is_blue_side else 0)
            enemy_champion_category_index_value = int(
                CATEGORY_TO_INDEX.get(str(enemy_champion_category_name).upper(), CATEGORY_UNKNOWN_INDEX)
            )
            row_count = int(numeric_array.shape[0])
            row_start = int(global_row_start)
            row_end = int(row_start + row_count)

            all_numeric_rows.append(numeric_array)
            all_row_type_rows.append(row_type_array)
            all_lane_type_rows.append(lane_type_array)
            all_target_region_rows.append(target_region_array)
            global_row_start = row_end

            metadata_handle.write(
                json.dumps(
                    {
                        "matchId": payload.get("matchId"),
                        "participantId": payload.get("participantId"),
                        "teamId": payload.get("teamId"),
                        "isBlueSide": is_blue_side,
                        "championName": payload.get("championName"),
                        "enemyChampionName": payload.get("enemyChampionName"),
                        "sourcePath": str(path),
                        "datasetPath": "dataset.npz",
                        "rowStart": row_start,
                        "rowEnd": row_end,
                        "rows": len(rows),
                        "validSamples": len(valid_sample_indices),
                        "validSampleIndices": valid_indices_array.tolist(),
                        "sideIndex": side_index_value,
                        "enemyChampionCategoryIndex": enemy_champion_category_index_value,
                        "rowTypeCounts": dict(shard_row_type_counts),
                        "regionCounts": dict(shard_region_counts),
                    },
                    separators=(",", ":"),
                )
            )
            metadata_handle.write("\n")

            processed_paths += 1
            now = time.time()
            should_print = (
                processed_paths == len(source_paths)
                or processed_paths - last_progress_count >= PROGRESS_PRINT_EVERY_COUNT
                or now - last_progress_time >= PROGRESS_PRINT_EVERY_SECONDS
            )
            if should_print:
                print(
                    f"[loadData] champion={champion or 'ALL'} {format_progress_bar(processed_paths, len(source_paths))}",
                    flush=True,
                )
                last_progress_count = processed_paths
                last_progress_time = now

    numeric_payload = (
        np.concatenate(all_numeric_rows, axis=0)
        if all_numeric_rows
        else np.zeros((0, len(NUMERIC_FEATURE_NAMES)), dtype=np.float32)
    )
    row_type_payload = (
        np.concatenate(all_row_type_rows, axis=0)
        if all_row_type_rows
        else np.zeros((0,), dtype=np.int16)
    )
    lane_type_payload = (
        np.concatenate(all_lane_type_rows, axis=0)
        if all_lane_type_rows
        else np.zeros((0,), dtype=np.int16)
    )
    target_region_payload = (
        np.concatenate(all_target_region_rows, axis=0)
        if all_target_region_rows
        else np.zeros((0,), dtype=np.int16)
    )

    np.savez_compressed(
        paths["dataset"],
        numeric=numeric_payload,
        rowType=row_type_payload,
        laneType=lane_type_payload,
        targetRegion=target_region_payload,
    )

    paths["feature_schema"].write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    summary_payload = {
        "logRoot": str(source_root),
        "champion": champion,
        "filesScanned": len(source_paths),
        "sequenceFiles": len(source_paths),
        "rows": total_rows,
        "samples": total_valid_samples,
        "windowSize": int(window_size),
        "numericFeatureCount": len(NUMERIC_FEATURE_NAMES),
        "targetType": "region_classification_sequence",
        "rowTypeCounts": dict(summary_row_type_counts),
        "regionCounts": dict(summary_region_counts),
        "skippedCounts": dict(skipped_counts),
        "outputDir": str(paths["input_dir"]),
        "savedArtifacts": {
            "featureSchema": "feature_schema.json",
            "sampleMetadata": "sequence_metadata.jsonl",
            "dataset": "dataset.npz",
        },
    }
    paths["summary"].write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return paths["input_dir"]


def main() -> None:
    output_dir = export_dataset(
        log_root=LOG_ROOT,
        champion=CHAMPION,
        input_root=INPUT_ROOT,
        window_size=WINDOW_SIZE,
        limit_files=MAX_FILES,
        include_row_types=None,
    )
    summary = load_saved_summary(champion=CHAMPION, input_root=INPUT_ROOT)
    print(json.dumps(summary, indent=2))
    print(f"Saved input dataset to {output_dir}")


if __name__ == "__main__":
    main()

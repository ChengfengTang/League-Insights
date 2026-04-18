"""
Copy processed match logs under Data/log/ into champion-named folders for the two junglers.

Each log is one JSON object with matchContext (blueJungler / redJungler), junglerTrainingRows,
and events — same shape as processMatch.py output (`<matchId>.jsonl` under each run).

Writes:
  Data/log (champions)/<championName>/<matchId>.jsonl

Only the champion folder groups files; run is not part of the output path. Reads only flat
`Data/log/<run>/*.jsonl`. Skips when the destination file already exists.
Single-threaded.

Example:
  python3 Data/sortLog.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

dataDir = Path(__file__).resolve().parent
logRootDir = dataDir / "log"
championsLogRootDir = dataDir / "log (champions)"


def safeChampionDirName(championName: str) -> str:
    if not championName or not str(championName).strip():
        return "unknown"
    name = str(championName).strip()
    cleaned = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    cleaned = cleaned.strip("_") or "unknown"
    return cleaned[:200]


def loadLogPayload(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def junglerChampionNames(matchContext: Dict) -> List[str]:
    names: List[str] = []
    for key in ("blueJungler", "redJungler"):
        block = matchContext.get(key) or {}
        name = block.get("championName")
        if name:
            names.append(str(name))
    return names


def discoverLogSourcesForRun(runDir: Path) -> List[Tuple[Path, str]]:
    """Sources directly under one run folder: only top-level *.jsonl."""
    return [(path, path.stem) for path in sorted(p for p in runDir.glob("*.jsonl") if p.is_file())]


def sortLogsIntoChampionFolders(*, dryRun: bool) -> None:
    if not logRootDir.is_dir():
        print(f"No log root: {logRootDir}", flush=True)
        return

    written = 0
    plannedWrites = 0
    skippedExisting = 0
    skippedNoJunglers = 0
    errors = 0

    for runDir in sorted(p for p in logRootDir.iterdir() if p.is_dir()):
        for sourcePath, matchId in discoverLogSourcesForRun(runDir):
            try:
                payload = loadLogPayload(sourcePath)
            except (OSError, json.JSONDecodeError) as exc:
                errors += 1
                print(f"[error] {sourcePath}: {exc}", flush=True)
                continue

            matchContext = payload.get("matchContext") or {}
            champs = junglerChampionNames(matchContext)
            if not champs:
                skippedNoJunglers += 1
                continue

            seenDest: Set[Path] = set()
            for championName in champs:
                folder = championsLogRootDir / safeChampionDirName(championName)
                destPath = folder / f"{matchId}.jsonl"
                if destPath in seenDest:
                    continue
                seenDest.add(destPath)

                if destPath.exists():
                    skippedExisting += 1
                    continue

                if dryRun:
                    print(f"[dry-run] would write {destPath}", flush=True)
                    plannedWrites += 1
                    continue

                folder.mkdir(parents=True, exist_ok=True)
                destPath.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                written += 1

    if dryRun:
        print(
            f"Done (dry-run). planned={plannedWrites} skip_existing={skippedExisting} "
            f"skip_no_junglers={skippedNoJunglers} errors={errors}",
            flush=True,
        )
    else:
        print(
            f"Done. wrote={written} skip_existing={skippedExisting} "
            f"skip_no_junglers={skippedNoJunglers} errors={errors}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort Data/log into Data/log (champions)/ by jungler champion.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files.",
    )
    args = parser.parse_args()
    sortLogsIntoChampionFolders(dryRun=args.dry_run)


if __name__ == "__main__":
    main()

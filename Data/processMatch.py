"""
Process raw match and timeline JSON into ML-ready per-jungler files.

Reads flat `Data/matches/<run>/*.json` with `Data/timelines/<run>/<matchId>_timeline.json`.
Writes one pretty-printed JSON file per jungler, grouped under champion folders:
  Data/log/<run>/<ChampionName>/<matchId>_p<participantId>_<blue|red>.json

Each file has a header (constant per file) and a `rows` array (one per frame during
the laning phase, default first 14 minutes). Rows are the minimal leakage-safe
feature set used for enemy-jungler +60s location prediction.

Example:
  python3 Data/processMatch.py na1-ranked-solo-5x5-20260413-235836
"""


from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

dataDir = Path(__file__).resolve().parent
matchesDir = dataDir / "matches"
timelinesDir = dataDir / "timelines"
logRootDir = dataDir / "log"

workerShutdown = None
DEFAULT_TARGET_HORIZON_MS = 60000
TARGET_HORIZON_MIN_MS = 50000
TARGET_HORIZON_MAX_MS = 70000
LANING_PHASE_MS = 14 * 60 * 1000
# Approximate own-fountain coords on Summoner's Rift, per team.
BASE_COORDS_BY_TEAM = {100: (1500, 1500), 200: (13500, 13500)}
# If a jungler is within this many units of their own fountain at frame i,
# we treat an item purchase as a starting purchase, not a recall/back.
BACKING_MIN_DEPARTURE_DISTANCE = 1800.0
# Collapse consecutive purchases from the same shop visit into one backing row.
BACKING_PURCHASE_MERGE_WINDOW_MS = 15000
# Approximate early-game respawn timers by level; good enough for laning-phase
# sequencing when we want death -> respawn/base -> next frame in the logs.
RESPAWN_SECONDS_BY_LEVEL = {
    1: 10,
    2: 10,
    3: 12,
    4: 12,
    5: 14,
    6: 16,
    7: 20,
    8: 25,
    9: 28,
    10: 32,
    11: 35,
    12: 40,
    13: 45,
    14: 50,
    15: 52,
    16: 55,
    17: 58,
    18: 60,
}
PROGRESS_BAR_WIDTH = 30
PROGRESS_PRINT_EVERY_COUNT = 100
PROGRESS_PRINT_EVERY_SECONDS = 10.0


def formatProgressBar(done: int, total: int) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int((done / total) * PROGRESS_BAR_WIDTH)
    bar = "=" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    return f"[{bar}] {done}/{total}"


def resolveMatchesRunDir(runName: str) -> Path:
    """Data/matches/<basename(runName)>/ must exist."""
    name = Path(runName.strip()).name
    if not name:
        raise RuntimeError("Run name is empty.")
    root = (matchesDir / name).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Matches run folder not found: {root}")
    return root


def workerProcessCount(numChunks: int) -> int:
    """Run up to one process per match chunk (each chunk is a list of match files)."""
    if numChunks <= 0:
        return 0
    return max(1, numChunks)


def chunkMatchPaths(matchPaths: List[Path], numChunks: int) -> List[List[Path]]:
    if not matchPaths:
        return []
    numChunks = max(1, min(numChunks, len(matchPaths)))
    buckets: List[List[Path]] = [[] for _ in range(numChunks)]
    for index, path in enumerate(matchPaths):
        buckets[index % numChunks].append(path)
    return buckets


def distanceBetween(x1: Optional[float], y1: Optional[float], x2: Optional[float], y2: Optional[float]) -> Optional[float]:
    if None in (x1, y1, x2, y2):
        return None
    return math.hypot(x2 - x1, y2 - y1)


def buildStaticMatchContext(matchMeta: Dict) -> Dict:
    info = matchMeta.get("info", {})
    participantsRaw = info.get("participants", [])
    teamsRaw = info.get("teams", [])

    participants: List[Dict] = []
    blueJungler: Dict = {}
    redJungler: Dict = {}

    for participant in participantsRaw:
        entry = {
            "participantId": participant.get("participantId"),
            "teamId": participant.get("teamId"),
            "championId": participant.get("championId"),
            "championName": participant.get("championName"),
            "teamPosition": participant.get("teamPosition"),
            "individualPosition": participant.get("individualPosition"),
            "lane": participant.get("lane"),
            "role": participant.get("role"),
            "summoner1Id": participant.get("summoner1Id"),
            "summoner2Id": participant.get("summoner2Id"),
        }
        participants.append(entry)

        if participant.get("teamPosition") == "JUNGLE":
            jungleEntry = {
                "participantId": participant.get("participantId"),
                "championId": participant.get("championId"),
                "championName": participant.get("championName"),
            }
            if participant.get("teamId") == 100:
                blueJungler = jungleEntry
            elif participant.get("teamId") == 200:
                redJungler = jungleEntry

    teams: List[Dict] = []
    for team in teamsRaw:
        teams.append(
            {
                "teamId": team.get("teamId"),
                "win": team.get("win"),
                "objectives": team.get("objectives", {}),
            }
        )

    return {
        "rowType": "static_match_context",
        "matchId": matchMeta.get("metadata", {}).get("matchId"),
        "gameVersion": info.get("gameVersion"),
        "queueId": info.get("queueId"),
        "mapId": info.get("mapId"),
        "gameDuration": info.get("gameDuration"),
        "gameStartTimestamp": info.get("gameStartTimestamp"),
        "teams": teams,
        "participants": participants,
        "blueJungler": blueJungler,
        "redJungler": redJungler,
    }


def buildParticipantIndex(matchMeta: Dict) -> Dict[int, Dict]:
    byId: Dict[int, Dict] = {}
    for participant in matchMeta.get("info", {}).get("participants", []):
        participantId = participant.get("participantId")
        if isinstance(participantId, int):
            byId[participantId] = participant
    return byId


def buildTimelineData(frames: List[Dict]) -> Dict[int, List[Dict]]:
    participantData: Dict[int, List[Dict]] = {i: [] for i in range(1, 11)}
    for frame in frames:
        timestampMs = frame["timestamp"]
        for playerKey, participantFrame in frame["participantFrames"].items():
            if "position" not in participantFrame:
                continue
            championStats = participantFrame.get("championStats", {})
            participantId = int(playerKey)
            row = {
                "frameIndex": len(participantData[participantId]),
                "timestampMs": timestampMs,
                "x": participantFrame["position"]["x"],
                "y": participantFrame["position"]["y"],
                "level": participantFrame.get("level", 1),
                "xp": participantFrame.get("xp", 0),
                "currentGold": participantFrame.get("currentGold", 0),
                "totalGold": participantFrame.get("totalGold", 0),
                "csJungle": participantFrame.get("jungleMinionsKilled", 0),
                "csLane": participantFrame.get("minionsKilled", 0),
                "hp": championStats.get("health", 0),
                "hpMax": championStats.get("healthMax", 0),
                "movementSpeed": championStats.get("movementSpeed", 0),
                "timeEnemySpentControlled": participantFrame.get("timeEnemySpentControlled", 0),
            }
            participantData[participantId].append(row)
    return participantData


def buildEventData(frames: List[Dict]) -> List[Dict]:
    events: List[Dict] = []
    for frame in frames:
        for event in frame.get("events", []):
            timestampMs = event["timestamp"]
            eventType = event["type"]
            position = event.get("position") or {}
            if eventType == "CHAMPION_KILL":
                eventRow = {
                    "timestampMs": timestampMs,
                    "type": eventType,
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "killerId": event.get("killerId"),
                    "victim": event.get("victimId"),
                    "assists": event.get("assistingParticipantIds", []),
                }
            elif eventType == "ELITE_MONSTER_KILL":
                eventRow = {
                    "timestampMs": timestampMs,
                    "type": eventType,
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "killerId": event.get("killerId"),
                    "monster": event.get("monsterType"),
                    "monsterSubType": event.get("monsterSubType"),
                    "assists": event.get("assistingParticipantIds", []),
                }
            elif eventType == "TURRET_PLATE_DESTROYED":
                eventRow = {
                    "timestampMs": timestampMs,
                    "type": eventType,
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "killerId": event.get("killerId"),
                    "laneType": event.get("laneType"),
                    "teamId": event.get("teamId"),
                }
            elif eventType == "ITEM_PURCHASED":
                eventRow = {
                    "timestampMs": timestampMs,
                    "type": eventType,
                    "participantId": event.get("participantId"),
                    "itemId": event.get("itemId"),
                }
            else:
                continue

            events.append(eventRow)
    return events


def getJunglerIds(staticMatchContext: Dict) -> List[int]:
    junglerIds: List[int] = []
    for jungler in (staticMatchContext.get("blueJungler", {}), staticMatchContext.get("redJungler", {})):
        participantId = jungler.get("participantId")
        if isinstance(participantId, int):
            junglerIds.append(participantId)
    return junglerIds


def safeChampionDirName(name: Optional[str]) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name or ""))
    return cleaned or "unknown"


ROW_TYPE_SORT_ORDER = {
    "frame": 0,
    "kill": 1,
    "assist": 1,
    "dragon": 1,
    "plate": 1,
    "backing": 2,
    "death": 3,
    "respawn": 4,
}


def _lookupFrameAtOrBefore(timelineRows: List[Dict], timestampMs: int) -> Optional[Dict]:
    """Return the last frame row with timestampMs <= target, or None."""
    match: Optional[Dict] = None
    for row in timelineRows:
        if row["timestampMs"] > timestampMs:
            break
        match = row
    return match


def _inferRespawnSeconds(level: Optional[int]) -> int:
    if not isinstance(level, int):
        return 15
    return RESPAWN_SECONDS_BY_LEVEL.get(level, RESPAWN_SECONDS_BY_LEVEL[18])


def _fillSequentialTargetsAndFrameIndices(rows: List[Dict]) -> List[Dict]:
    """
    After rows are merged and time-sorted, assign a unique sequential frameIndex
    and point each row's target fields at the next row in the ordered sequence.
    """
    for idx, row in enumerate(rows):
        row["frameIndex"] = idx
        nextRow = rows[idx + 1] if idx + 1 < len(rows) else None
        if nextRow is None:
            row["targetHorizonMs"] = None
            row["targetX"] = None
            row["targetY"] = None
            row["targetDx"] = None
            row["targetDy"] = None
            continue

        row["targetHorizonMs"] = nextRow["timestampMs"] - row["timestampMs"]
        row["targetX"] = nextRow.get("x")
        row["targetY"] = nextRow.get("y")

        currentX = row.get("x")
        currentY = row.get("y")
        nextX = nextRow.get("x")
        nextY = nextRow.get("y")
        row["targetDx"] = (nextX - currentX) if currentX is not None and nextX is not None else None
        row["targetDy"] = (nextY - currentY) if currentY is not None and nextY is not None else None

    return rows


def buildJunglerEventRows(
    participantId: int,
    teamId: Optional[int],
    allEvents: List[Dict],
    timelineRows: List[Dict],
    laningPhaseMs: int,
) -> List[Dict]:
    """
    Emit per-event rows for this jungler: kills, assists, deaths, dragons,
    plates, purchase-driven backings, and respawns. Deaths also emit a
    respawn/base row with an inferred timestamp so merged rows can form chains
    like death -> respawn -> frame.
    """
    out: List[Dict] = []
    deathTimestamps: List[int] = []
    respawnTimestamps: List[int] = []
    lastBackingTs: Optional[int] = None

    def context(ts: int) -> Dict:
        frame = _lookupFrameAtOrBefore(timelineRows, ts)
        return {
            "frameIndex": frame.get("frameIndex") if frame else None,
            "level": frame.get("level") if frame else None,
        }

    for event in allEvents:
        ts = event.get("timestampMs")
        if ts is None or ts > laningPhaseMs:
            continue

        etype = event.get("type")
        killerId = event.get("killerId")
        victimId = event.get("victim")
        assists = event.get("assists") or []

        rowType: Optional[str] = None
        extras: Dict = {}

        if etype == "CHAMPION_KILL":
            if killerId == participantId:
                rowType = "kill"
            elif victimId == participantId:
                rowType = "death"
            elif participantId in assists:
                rowType = "assist"
        elif etype == "ELITE_MONSTER_KILL" and event.get("monster") == "DRAGON":
            if killerId == participantId or participantId in assists:
                rowType = "dragon"
                extras = {"monsterSubType": event.get("monsterSubType")}
        elif etype == "TURRET_PLATE_DESTROYED":
            if killerId == participantId:
                rowType = "plate"
                extras = {"laneType": event.get("laneType")}
        elif etype == "ITEM_PURCHASED":
            if event.get("participantId") == participantId:
                baseX, baseY = BASE_COORDS_BY_TEAM.get(teamId or 0, (1500, 1500))
                priorFrame = _lookupFrameAtOrBefore(timelineRows, ts)
                if priorFrame is None:
                    continue
                priorX = priorFrame.get("x")
                priorY = priorFrame.get("y")
                if priorX is None or priorY is None:
                    continue
                if math.hypot(priorX - baseX, priorY - baseY) < BACKING_MIN_DEPARTURE_DISTANCE:
                    continue
                if any(ts <= respawnTs <= ts + BACKING_PURCHASE_MERGE_WINDOW_MS for respawnTs in respawnTimestamps):
                    continue
                if lastBackingTs is not None and ts - lastBackingTs <= BACKING_PURCHASE_MERGE_WINDOW_MS:
                    continue
                rowType = "backing"
                lastBackingTs = ts

        if rowType is None:
            continue

        ctx = context(ts)
        eventX = event.get("x")
        eventY = event.get("y")
        out.append(
            {
                "rowType": rowType,
                "timestampMs": ts,
                "frameIndex": -1,
                "x": eventX,
                "y": eventY,
                "level": ctx["level"],
                **extras,
            }
        )

        if rowType == "death":
            deathTimestamps.append(ts)
            respawnTs = ts + (_inferRespawnSeconds(ctx["level"]) * 1000)
            if respawnTs <= laningPhaseMs:
                respawnTimestamps.append(respawnTs)
                baseX, baseY = BASE_COORDS_BY_TEAM.get(teamId or 0, (1500, 1500))
                out.append(
                    {
                        "rowType": "respawn",
                        "timestampMs": respawnTs,
                        "frameIndex": -1,
                        "x": baseX,
                        "y": baseY,
                        "level": ctx["level"],
                    }
                )

    return out


def buildJunglerGroupedPayloads(
    staticMatchContext: Dict,
    participantIndex: Dict[int, Dict],
    participantTimelineData: Dict[int, List[Dict]],
    events: List[Dict],
    targetHorizonMs: int = DEFAULT_TARGET_HORIZON_MS,
    laningPhaseMs: int = LANING_PHASE_MS,
) -> List[Dict]:
    """
    Build one payload per jungler (header + rows). Rows include time-sampled
    `frame` rows plus event rows (kill/death/respawn/dragon/plate/backing), all
    sorted by timestampMs ascending. After merging, targets are filled from each
    row to the very next row in the ordered sequence.
    """
    junglerIds = getJunglerIds(staticMatchContext)
    matchId = staticMatchContext.get("matchId")
    blueJungler = staticMatchContext.get("blueJungler") or {}
    redJungler = staticMatchContext.get("redJungler") or {}
    payloads: List[Dict] = []

    for participantId in junglerIds:
        participantMeta = participantIndex.get(participantId, {})
        teamId = participantMeta.get("teamId")
        isBlue = teamId == 100
        enemyJg = redJungler if isBlue else blueJungler
        timelineRows = participantTimelineData.get(participantId, [])

        frameRows: List[Dict] = []
        for row in timelineRows:
            timestampMs = row["timestampMs"]
            if timestampMs > laningPhaseMs:
                break

            frameRows.append(
                {
                    "rowType": "frame",
                    "timestampMs": timestampMs,
                    "frameIndex": -1,
                    "x": row["x"],
                    "y": row["y"],
                    "level": row["level"],
                    "csJungle": row["csJungle"],
                    "csLane": row["csLane"],
                    "currentGold": row["currentGold"],
                    "totalGold": row["totalGold"],
                    "movementSpeed": row["movementSpeed"],
                }
            )

        if not frameRows:
            continue

        eventRows = buildJunglerEventRows(
            participantId=participantId,
            teamId=teamId,
            allEvents=events,
            timelineRows=timelineRows,
            laningPhaseMs=laningPhaseMs,
        )

        mergedRows = sorted(
            frameRows + eventRows,
            key=lambda r: (r["timestampMs"], ROW_TYPE_SORT_ORDER.get(r["rowType"], 99)),
        )
        mergedRows = _fillSequentialTargetsAndFrameIndices(mergedRows)

        payloads.append(
            {
                "matchId": matchId,
                "participantId": participantId,
                "teamId": teamId,
                "isBlueSide": isBlue,
                "championId": participantMeta.get("championId"),
                "championName": participantMeta.get("championName"),
                "enemyChampionId": enemyJg.get("championId"),
                "enemyChampionName": enemyJg.get("championName"),
                "rows": mergedRows,
            }
        )

    return payloads


def processOneMatchFiles(matchPath: Path, timelinePath: Path, logRunDir: Path) -> bool:
    """
    Load a match/timeline pair and write one pretty JSON per jungler under
    `logRunDir/<ChampionName>/<matchId>_p<pid>_<side>.json`. Returns True if
    at least one jungler file was written (or the match had no valid jungler rows).
    """
    if not timelinePath.exists() or not matchPath.exists():
        return False

    timeline = json.loads(timelinePath.read_text(encoding="utf-8"))
    matchMeta = json.loads(matchPath.read_text(encoding="utf-8"))

    frames = timeline["info"]["frames"]
    staticMatchContext = buildStaticMatchContext(matchMeta)
    participantIndex = buildParticipantIndex(matchMeta)
    participantData = buildTimelineData(frames)
    allEvents = buildEventData(frames)
    payloads = buildJunglerGroupedPayloads(
        staticMatchContext=staticMatchContext,
        participantIndex=participantIndex,
        participantTimelineData=participantData,
        events=allEvents,
    )

    matchId = staticMatchContext.get("matchId") or matchPath.stem
    logRunDir.mkdir(parents=True, exist_ok=True)

    for payload in payloads:
        championDir = logRunDir / safeChampionDirName(payload.get("championName"))
        championDir.mkdir(parents=True, exist_ok=True)
        sideLabel = "blue" if payload.get("isBlueSide") else "red"
        outPath = championDir / f"{matchId}_p{payload['participantId']}_{sideLabel}.json"
        outPath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return True


def processMatchChunkJob(task: Dict) -> None:
    """Process a chunk of flat `matches/<run>/*.json` with paired timelines under `timelines/<run>/`."""
    runName = task["runName"]
    chunkLabel = task["chunkLabel"]
    dataDirPath = Path(task["dataDirStr"])
    timelinesRunDir = timelinesDir / runName
    logRunDir = logRootDir / runName
    doneDir = logRunDir / ".done"
    logRunDir.mkdir(parents=True, exist_ok=True)
    doneDir.mkdir(parents=True, exist_ok=True)

    matchPaths = [Path(p) for p in task["matchPaths"]]
    totalMatches = len(matchPaths)
    processedMatches = 0
    lastProgressCount = -1
    lastProgressTime = 0.0

    print(f"[{chunkLabel}] {formatProgressBar(0, max(1, totalMatches))}", flush=True)

    for matchPath in matchPaths:
        matchId = matchPath.stem
        timelinePath = timelinesRunDir / f"{matchId}_timeline.json"
        doneMarker = doneDir / matchId
        try:
            if doneMarker.exists():
                if matchPath.exists():
                    matchPath.unlink()
                if timelinePath.exists():
                    timelinePath.unlink()
                success = True
            else:
                success = processOneMatchFiles(matchPath, timelinePath, logRunDir)
                if success:
                    doneMarker.touch()
                    if matchPath.exists():
                        matchPath.unlink()
                    if timelinePath.exists():
                        timelinePath.unlink()
        except Exception as exc:
            success = False
            print(f"[{chunkLabel}] Failed {matchId}: {exc}", flush=True)

        if not success:
            print(f"[{chunkLabel}] Unsuccessful log for {matchId}", flush=True)

        processedMatches += 1
        now = time.time()
        shouldPrint = (
            processedMatches == totalMatches
            or processedMatches - lastProgressCount >= PROGRESS_PRINT_EVERY_COUNT
            or now - lastProgressTime >= PROGRESS_PRINT_EVERY_SECONDS
        )
        if shouldPrint:
            print(
                f"[{chunkLabel}] {formatProgressBar(processedMatches, totalMatches)}",
                flush=True,
            )
            lastProgressCount = processedMatches
            lastProgressTime = now


def buildMatchChunkTasks(matchesRunRoot: Path) -> List[Dict]:
    runName = matchesRunRoot.name
    dataDirStr = str(dataDir.resolve())
    matchFiles = sorted(matchesRunRoot.glob("*.json"))
    if not matchFiles:
        return []

    cpus = multiprocessing.cpu_count() or 4
    numChunks = max(1, min(cpus, len(matchFiles)))
    chunks = chunkMatchPaths(matchFiles, numChunks)

    tasks: List[Dict] = []
    for index, paths in enumerate(chunks):
        tasks.append(
            {
                "runName": runName,
                "chunkLabel": f"chunk-{index}",
                "matchPaths": [str(p.resolve()) for p in paths],
                "dataDirStr": dataDirStr,
            }
        )
    return tasks


def matchChunkWorkerLoop(
    taskQueue: "multiprocessing.Queue",
    doneQueue: "multiprocessing.Queue",
) -> None:
    while True:
        item = taskQueue.get()
        if item is workerShutdown:
            break
        task = item
        try:
            processMatchChunkJob(task)
            doneQueue.put(task["chunkLabel"])
        except Exception as exc:
            label = task.get("chunkLabel", "?") if isinstance(task, dict) else "?"
            print(f"[worker] failed job {label}: {exc}", flush=True)
            doneQueue.put(("__error__", str(exc)))


def runDynamicMatchChunkPool(tasks: List[Dict]) -> None:
    numJobs = len(tasks)
    numWorkers = workerProcessCount(numJobs)

    if numJobs == 0:
        print("No match JSON files found under the run folder.")
        return

    if numWorkers == 1:
        for task in tasks:
            processMatchChunkJob(task)
        return

    manager = multiprocessing.Manager()
    taskQueue = manager.Queue()
    doneQueue = manager.Queue()

    workers: List[multiprocessing.Process] = []
    for _ in range(numWorkers):
        proc = multiprocessing.Process(
            target=matchChunkWorkerLoop,
            args=(taskQueue, doneQueue),
        )
        proc.start()
        workers.append(proc)

    inFlight: Set[str] = set()
    nextJobIndex = 0

    def tryDispatch() -> None:
        nonlocal nextJobIndex
        while len(inFlight) < numWorkers and nextJobIndex < numJobs:
            task = tasks[nextJobIndex]
            nextJobIndex += 1
            taskQueue.put(task)
            inFlight.add(task["chunkLabel"])

    tryDispatch()

    while inFlight or nextJobIndex < numJobs:
        if not inFlight:
            if nextJobIndex >= numJobs:
                break
            tryDispatch()
            if not inFlight:
                continue

        finished = doneQueue.get()
        if isinstance(finished, tuple) and finished[0] == "__error__":
            for _ in workers:
                taskQueue.put(workerShutdown)
            raise RuntimeError(finished[1])
        inFlight.discard(finished)
        tryDispatch()

    for _ in workers:
        taskQueue.put(workerShutdown)
    for proc in workers:
        proc.join(timeout=600)
        if proc.is_alive():
            proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process flat match JSON under Data/matches/<runName>/ into Data/log/<runName>/."
    )
    parser.add_argument(
        "runName",
        help="Run folder name only, e.g. na1-ranked-solo-5x5-20260413-235836 (uses Data/matches/<runName>/)",
    )
    args = parser.parse_args()

    matchesRunRoot = resolveMatchesRunDir(args.runName)
    tasks = buildMatchChunkTasks(matchesRunRoot)

    print(f"Matches run: {matchesRunRoot}")
    if not tasks:
        print("No *.json match files found directly under the run folder.")
        return
    numWorkers = workerProcessCount(len(tasks))
    totalFiles = sum(len(t["matchPaths"]) for t in tasks)
    print(f"Chunks ({len(tasks)}), {totalFiles} matches, up to {numWorkers} worker processes")
    print(f"Log root: {logRootDir / matchesRunRoot.name}/<ChampionName>/<matchId>_p<pid>_<side>.json")

    runDynamicMatchChunkPool(tasks)
    shutil.rmtree(matchesRunRoot, ignore_errors=True)
    shutil.rmtree(timelinesDir / matchesRunRoot.name, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()

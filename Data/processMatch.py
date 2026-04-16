"""
Process raw match and timeline JSON into one ML-ready `<matchId>.jsonl` per match.

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
NEARBY_EVENT_RADIUS = 2500.0
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


def workerProcessCount(numJobs: int) -> int:
    """Use exactly one worker per division job."""
    if numJobs <= 0:
        return 0
    return max(1, numJobs)


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
            if eventType == "CHAMPION_KILL":
                eventRow = {
                    "timestampMs": timestampMs,
                    "type": eventType,
                    "x": event.get("position", {}).get("x"),
                    "y": event.get("position", {}).get("y"),
                    "killerId": event.get("killerId"),
                    "victim": event.get("victimId"),
                    "assists": event.get("assistingParticipantIds", []),
                }
            elif eventType == "ELITE_MONSTER_KILL":
                eventRow = {
                    "timestampMs": timestampMs,
                    "type": eventType,
                    "x": event.get("position", {}).get("x"),
                    "y": event.get("position", {}).get("y"),
                    "killerId": event.get("killerId"),
                    "monster": event.get("monsterType"),
                    "monsterSubType": event.get("monsterSubType"),
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


def filterEventsForJunglers(events: List[Dict], junglerIds: List[int]) -> List[Dict]:
    junglerIdSet = set(junglerIds)
    filtered: List[Dict] = []
    for event in events:
        if event.get("x") is None or event.get("y") is None:
            continue

        killerId = event.get("killerId")
        victimId = event.get("victim")
        assists = event.get("assists", [])
        if killerId in junglerIdSet or victimId in junglerIdSet or any(assistId in junglerIdSet for assistId in assists):
            filtered.append(event)
    return filtered


def buildRecentEventFeatures(
    events: List[Dict],
    participantRow: Dict,
    participantId: int,
    participantTeamId: int,
    timestampMs: int,
) -> Dict:
    recentKillCount = 0
    recentObjectiveCount = 0
    recentDeathNearby = 0
    nearestKillDistance: Optional[float] = None
    nearestObjectiveDistance: Optional[float] = None
    timeSinceLastDragonMs: Optional[int] = None
    timeSinceLastHeraldMs: Optional[int] = None
    timeSinceLastBaronMs: Optional[int] = None

    x = participantRow["x"]
    y = participantRow["y"]

    for event in events:
        eventTs = event["timestampMs"]
        if eventTs > timestampMs:
            break
        ageMs = timestampMs - eventTs

        if event["type"] == "CHAMPION_KILL" and ageMs <= 60000:
            distance = distanceBetween(x, y, event.get("x"), event.get("y"))
            if distance is not None and distance <= NEARBY_EVENT_RADIUS:
                recentKillCount += 1
                nearestKillDistance = distance if nearestKillDistance is None else min(nearestKillDistance, distance)
                if event.get("victim") == participantId:
                    recentDeathNearby += 1

        if event["type"] == "ELITE_MONSTER_KILL":
            monster = event.get("monster")
            if monster == "DRAGON":
                timeSinceLastDragonMs = ageMs
            elif monster == "RIFTHERALD":
                timeSinceLastHeraldMs = ageMs
            elif monster == "BARON_NASHOR":
                timeSinceLastBaronMs = ageMs

            if ageMs <= 90000:
                distance = distanceBetween(x, y, event.get("x"), event.get("y"))
                if distance is not None and distance <= 5000.0:
                    recentObjectiveCount += 1
                    nearestObjectiveDistance = (
                        distance if nearestObjectiveDistance is None else min(nearestObjectiveDistance, distance)
                    )

    return {
        "nearbyChampionKillsLast60s": recentKillCount,
        "nearbyDeathsLast60s": recentDeathNearby,
        "nearbyObjectiveEventsLast90s": recentObjectiveCount,
        "nearestChampionKillDistance": nearestKillDistance,
        "nearestObjectiveDistance": nearestObjectiveDistance,
        "timeSinceLastDragonMs": timeSinceLastDragonMs,
        "timeSinceLastHeraldMs": timeSinceLastHeraldMs,
        "timeSinceLastBaronMs": timeSinceLastBaronMs,
        "isBlueSide": participantTeamId == 100,
    }


def buildJunglerTrainingRows(
    staticMatchContext: Dict,
    participantIndex: Dict[int, Dict],
    participantTimelineData: Dict[int, List[Dict]],
    events: List[Dict],
    targetHorizonMs: int = DEFAULT_TARGET_HORIZON_MS,
) -> List[Dict]:
    junglerIds = getJunglerIds(staticMatchContext)

    rows: List[Dict] = []
    for participantId in junglerIds:
        participantMeta = participantIndex.get(participantId, {})
        timelineRows = participantTimelineData.get(participantId, [])
        for idx, row in enumerate(timelineRows):
            previousRow = timelineRows[idx - 1] if idx > 0 else None
            nextRow = timelineRows[idx + 1] if idx + 1 < len(timelineRows) else None

            hp = row.get("hp", 0)
            hpMax = row.get("hpMax", 0)
            hpRatio = None if not hpMax else round(hp / hpMax, 6)

            dx1 = None
            dy1 = None
            distance1 = None
            speedPerSec1 = None
            timeDeltaMs1 = None
            if previousRow is not None:
                dx1 = row["x"] - previousRow["x"]
                dy1 = row["y"] - previousRow["y"]
                distance1 = round(math.hypot(dx1, dy1), 3)
                timeDeltaMs1 = row["timestampMs"] - previousRow["timestampMs"]
                if timeDeltaMs1 > 0:
                    speedPerSec1 = round(distance1 / (timeDeltaMs1 / 1000.0), 6)

            targetValid = False
            targetX = None
            targetY = None
            targetDx = None
            targetDy = None
            if nextRow is not None and nextRow["timestampMs"] - row["timestampMs"] == targetHorizonMs:
                targetValid = True
                targetX = nextRow["x"]
                targetY = nextRow["y"]
                targetDx = nextRow["x"] - row["x"]
                targetDy = nextRow["y"] - row["y"]

            eventFeatures = buildRecentEventFeatures(
                events=events,
                participantRow=row,
                participantId=participantId,
                participantTeamId=participantMeta.get("teamId", 0),
                timestampMs=row["timestampMs"],
            )

            rows.append(
                {
                    "rowType": "jungler_frame",
                    "matchId": staticMatchContext.get("matchId"),
                    "gameVersion": staticMatchContext.get("gameVersion"),
                    "queueId": staticMatchContext.get("queueId"),
                    "mapId": staticMatchContext.get("mapId"),
                    "gameDuration": staticMatchContext.get("gameDuration"),
                    "participantId": participantId,
                    "teamId": participantMeta.get("teamId"),
                    "championId": participantMeta.get("championId"),
                    "championName": participantMeta.get("championName"),
                    "teamPosition": participantMeta.get("teamPosition"),
                    "individualPosition": participantMeta.get("individualPosition"),
                    "summoner1Id": participantMeta.get("summoner1Id"),
                    "summoner2Id": participantMeta.get("summoner2Id"),
                    "frameIndex": row["frameIndex"],
                    "timestampMs": row["timestampMs"],
                    "x": row["x"],
                    "y": row["y"],
                    "level": row["level"],
                    "xp": row["xp"],
                    "currentGold": row["currentGold"],
                    "totalGold": row["totalGold"],
                    "csJungle": row["csJungle"],
                    "csLane": row["csLane"],
                    "hp": hp,
                    "hpMax": hpMax,
                    "hpRatio": hpRatio,
                    "movementSpeed": row["movementSpeed"],
                    "timeEnemySpentControlled": row["timeEnemySpentControlled"],
                    "dx1": dx1,
                    "dy1": dy1,
                    "distance1": distance1,
                    "speedPerSec1": speedPerSec1,
                    "timeDeltaMs1": timeDeltaMs1,
                    "targetHorizonMs": targetHorizonMs,
                    "targetValid": targetValid,
                    "targetX": targetX,
                    "targetY": targetY,
                    "targetDx": targetDx,
                    "targetDy": targetDy,
                    **eventFeatures,
                }
            )

    return rows


def processOneMatchFiles(matchPath: Path, timelinePath: Path, outputPath: Path) -> bool:
    """Load pair of JSON files and write one ML-ready JSON file. Returns True if processed."""
    if not timelinePath.exists() or not matchPath.exists():
        return False

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    timeline = json.loads(timelinePath.read_text(encoding="utf-8"))
    matchMeta = json.loads(matchPath.read_text(encoding="utf-8"))

    frames = timeline["info"]["frames"]
    staticMatchContext = buildStaticMatchContext(matchMeta)
    participantIndex = buildParticipantIndex(matchMeta)
    participantData = buildTimelineData(frames)
    allEvents = buildEventData(frames)
    junglerEvents = filterEventsForJunglers(allEvents, getJunglerIds(staticMatchContext))
    trainingRows = buildJunglerTrainingRows(
        staticMatchContext=staticMatchContext,
        participantIndex=participantIndex,
        participantTimelineData=participantData,
        events=allEvents,
    )

    parsedData = {
        "matchContext": staticMatchContext,
        "junglerTrainingRows": trainingRows,
        "events": junglerEvents,
    }

    outputPath.write_text(json.dumps(parsedData, indent=2), encoding="utf-8")
    return True


def processDivisionJob(task: Dict) -> None:
    """All *.json matches under one division folder (paired with timelines/<run>/<division>/)."""
    runName = task["runName"]
    divisionLabel = task["divisionLabel"]
    matchesDivisionDir = Path(task["matchesDivisionDir"])
    timelinesDivisionDir = Path(task["timelinesDivisionDir"])
    dataDirPath = Path(task["dataDirStr"])

    matchFiles = sorted(matchesDivisionDir.glob("*.json"))
    totalMatches = len(matchFiles)
    processedMatches = 0
    lastProgressCount = -1
    lastProgressTime = 0.0

    print(f"[{divisionLabel}] {formatProgressBar(0, max(1, totalMatches))}", flush=True)

    for matchPath in matchFiles:
        matchId = matchPath.stem
        timelinePath = timelinesDivisionDir / f"{matchId}_timeline.json"
        outputPath = dataDirPath / "log" / runName / divisionLabel / f"{matchId}.jsonl"
        try:
            if outputPath.exists():
                # Already logged; clean up raw files if they still exist.
                if matchPath.exists():
                    matchPath.unlink()
                if timelinePath.exists():
                    timelinePath.unlink()
                success = True
            else:
                success = processOneMatchFiles(matchPath, timelinePath, outputPath)
                if success:
                    # Delete raw inputs after successful logging.
                    if matchPath.exists():
                        matchPath.unlink()
                    if timelinePath.exists():
                        timelinePath.unlink()
        except Exception as exc:
            success = False
            print(f"[{divisionLabel}] Failed {matchId}: {exc}", flush=True)

        if not success:
            print(f"[{divisionLabel}] Unsuccessful log for {matchId}", flush=True)

        processedMatches += 1
        now = time.time()
        shouldPrint = (
            processedMatches == totalMatches
            or processedMatches - lastProgressCount >= PROGRESS_PRINT_EVERY_COUNT
            or now - lastProgressTime >= PROGRESS_PRINT_EVERY_SECONDS
        )
        if shouldPrint:
            print(
                f"[{divisionLabel}] {formatProgressBar(processedMatches, totalMatches)}",
                flush=True,
            )
            lastProgressCount = processedMatches
            lastProgressTime = now

def buildDivisionTasks(matchesRunRoot: Path) -> List[Dict]:
    runName = matchesRunRoot.name
    dataDirStr = str(dataDir.resolve())
    tasks: List[Dict] = []

    for divisionDir in sorted(matchesRunRoot.iterdir()):
        if not divisionDir.is_dir():
            continue
        divisionLabel = divisionDir.name
        if not any(divisionDir.glob("*.json")):
            continue
        tasks.append(
            {
                "runName": runName,
                "divisionLabel": divisionLabel,
                "matchesDivisionDir": str(divisionDir.resolve()),
                "timelinesDivisionDir": str((timelinesDir / runName / divisionLabel).resolve()),
                "dataDirStr": dataDirStr,
            }
        )
    return tasks


def divisionWorkerLoop(
    taskQueue: "multiprocessing.Queue",
    doneQueue: "multiprocessing.Queue",
) -> None:
    while True:
        item = taskQueue.get()
        if item is workerShutdown:
            break
        task = item
        try:
            processDivisionJob(task)
            doneQueue.put(task["divisionLabel"])
        except Exception as exc:
            label = task.get("divisionLabel", "?") if isinstance(task, dict) else "?"
            print(f"[worker] failed job {label}: {exc}", flush=True)
            doneQueue.put(("__error__", str(exc)))


def runDynamicDivisionPool(tasks: List[Dict]) -> None:
    numJobs = len(tasks)
    numWorkers = workerProcessCount(numJobs)

    if numJobs == 0:
        print("No division folders with match JSON found.")
        return

    if numWorkers == 1:
        for task in tasks:
            processDivisionJob(task)
        return

    manager = multiprocessing.Manager()
    taskQueue = manager.Queue()
    doneQueue = manager.Queue()

    workers: List[multiprocessing.Process] = []
    for _ in range(numWorkers):
        proc = multiprocessing.Process(
            target=divisionWorkerLoop,
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
            inFlight.add(task["divisionLabel"])

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
        description="Process match JSON into logs for one fetchMatch run under Data/matches/<runName>/."
    )
    parser.add_argument(
        "runName",
        help="Run folder name only, e.g. na1-ranked-solo-5x5-20260413-235836 (uses Data/matches/<runName>/)",
    )
    args = parser.parse_args()

    matchesRunRoot = resolveMatchesRunDir(args.runName)
    tasks = buildDivisionTasks(matchesRunRoot)

    print(f"Matches run: {matchesRunRoot}")
    if not tasks:
        print("No division folders with match JSON found.")
        return
    numWorkers = workerProcessCount(len(tasks))
    print(f"Jobs ({len(tasks)}): " + " → ".join(t["divisionLabel"] for t in tasks))
    print(f"Worker processes: {numWorkers} (one per division)")
    print(f"Log root: {logRootDir / matchesRunRoot.name}/<division>/<matchId>.jsonl")

    runDynamicDivisionPool(tasks)
    shutil.rmtree(matchesRunRoot, ignore_errors=True)
    shutil.rmtree(timelinesDir / matchesRunRoot.name, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()

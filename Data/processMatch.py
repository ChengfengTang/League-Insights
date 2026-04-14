"""
Process raw match and timeline JSON into logs and parsedData under Data/log/.

No Riot API calls: work is local JSON only. Uses up to one worker process per CPU core (capped by
the number of division jobs) so divisions are processed in parallel as fast as this machine allows.

Example:
  python Data/processMatch.py na1-ranked-solo-5x5-20260413-235836

You pass only the fetchMatch run folder name. The script reads Data/matches/<runName>/ and pairs
each division subfolder with Data/timelines/<runName>/<division>/.

Expects the same on-disk layout as fetchMatch: under Data/matches/<run>/ one subfolder per
division, each with *.json match files, paired with Data/timelines/<run>/<division>/*. Each
division folder is one job; a small scheduler keeps up to N workers busy until all divisions
finish (N = min(job count, CPU count)).

Writes:
  Data/log/<run-folder-name>/<division-stem>/<matchId>/events.log
  Data/log/<run-folder-name>/<division-stem>/<matchId>/timelines.log
  Data/log/<run-folder-name>/<division-stem>/<matchId>/parsedData.json

CLI: one positional argument, the run folder name (e.g. na1-ranked-solo-5x5-20260413-235836).
"""


#Todo: If user's gold reduced, they backed, add that into consideration
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Set

dataDir = Path(__file__).resolve().parent
matchesDir = dataDir / "matches"
timelinesDir = dataDir / "timelines"
logRootDir = dataDir / "log"

workerShutdown = None


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
    """Use all logical CPUs, but never more processes than division jobs."""
    if numJobs <= 0:
        return 0
    cpus = os.cpu_count() or 1
    return max(1, min(numJobs, cpus))


def msToMinSec(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"


def calculateDeathTimer(level: int, gameMinutes: float) -> int:
    baseRespawnWindow = [-1, 10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5]
    clampedLevel = min(max(level, 1), 18)
    baseTimer = baseRespawnWindow[clampedLevel]

    if gameMinutes < 15:
        timeImpactFactor = 0
    elif gameMinutes < 30:
        timeImpactFactor = math.ceil(2 * (gameMinutes - 15)) * 0.425 / 100
    elif gameMinutes < 45:
        timeImpactFactor = 12.75 / 60 + math.ceil(2 * (gameMinutes - 30)) * 0.30 / 100
    else:
        timeImpactFactor = 21.75 / 60 + math.ceil(2 * (gameMinutes - 45)) * 1.45 / 100

    totalTimer = baseTimer * (1 + timeImpactFactor)
    return round(totalTimer)


def getChampLabel(playerId: int, championMap: Dict[str, Dict[str, str]]) -> str:
    playerKey = str(playerId)
    champion = championMap.get(playerKey, {}).get("champion", f"Champion {playerKey}")
    team = championMap.get(playerKey, {}).get("team", "Unknown")
    if team == "Blue":
        return f"Blue {champion}"
    return f"Red {champion}"


def buildChampionMap(matchMeta: Dict) -> Dict[str, Dict[str, str]]:
    championMap: Dict[str, Dict[str, str]] = {}
    for participant in matchMeta["info"]["participants"]:
        playerKey = str(participant["participantId"])
        championMap[playerKey] = {
            "champion": participant["championName"],
            "team": "Blue" if participant["teamId"] == 100 else "Red",
        }
    return championMap


def buildTimelineData(frames: List[Dict]) -> Dict[str, List[Dict]]:
    participantData: Dict[str, List[Dict]] = {str(i): [] for i in range(1, 11)}
    for frame in frames:
        timestampMs = frame["timestamp"]
        timeLabel = msToMinSec(timestampMs)
        for playerKey, participantFrame in frame["participantFrames"].items():
            if "position" not in participantFrame:
                continue
            row = {
                "timestampMs": timestampMs,
                "time": timeLabel,
                "x": participantFrame["position"]["x"],
                "y": participantFrame["position"]["y"],
                "level": participantFrame.get("level", 1),
                "cs": participantFrame.get("jungleMinionsKilled", 0) + participantFrame.get("minionsKilled", 0),
                "gold": participantFrame.get("currentGold", 0),
            }
            participantData[playerKey].append(row)
    return participantData


def buildEventData(frames: List[Dict]) -> List[Dict]:
    events: List[Dict] = []
    for frame in frames:
        for event in frame.get("events", []):
            timestampMs = event["timestamp"]
            timeLabel = msToMinSec(timestampMs)
            eventType = event["type"]
            eventRow = {"timestampMs": timestampMs, "time": timeLabel, "type": eventType}

            if eventType == "CHAMPION_KILL":
                eventRow.update(
                    {
                        "actor": event.get("killerId"),
                        "victim": event.get("victimId"),
                        "x": event.get("position", {}).get("x"),
                        "y": event.get("position", {}).get("y"),
                        "assists": event.get("assistingParticipantIds", []),
                    }
                )
            elif eventType == "ELITE_MONSTER_KILL":
                eventRow.update(
                    {
                        "actor": event.get("killerId"),
                        "monster": event.get("monsterType"),
                        "x": event.get("position", {}).get("x"),
                        "y": event.get("position", {}).get("y"),
                    }
                )
            elif eventType == "LEVEL_UP":
                eventRow.update({"actor": event.get("participantId"), "level": event.get("level")})
            else:
                continue

            events.append(eventRow)
    return events


def buildEventLines(events: List[Dict], championMap: Dict[str, Dict[str, str]]) -> List[str]:
    lines: List[str] = []
    levelByPlayer = {str(i): 1 for i in range(1, 11)}

    for event in events:
        timeLabel = event["time"]
        eventType = event["type"]

        if eventType == "CHAMPION_KILL":
            killer = event.get("actor")
            victim = event.get("victim")
            xValue = event.get("x", "?")
            yValue = event.get("y", "?")
            killerLabel = getChampLabel(killer, championMap)
            victimLabel = getChampLabel(victim, championMap)
            lines.append(f"{timeLabel} - {killerLabel} killed {victimLabel} at ({xValue}, {yValue})")

            for assistPlayerId in event.get("assists", []):
                assistLabel = getChampLabel(assistPlayerId, championMap)
                lines.append(f"{timeLabel} - {assistLabel} assisted at ({xValue}, {yValue})")

            victimLevel = levelByPlayer.get(str(victim), 1)
            deathTimer = calculateDeathTimer(victimLevel, int(timeLabel.split(":")[0]))
            nowSec = int(timeLabel.split(":")[0]) * 60 + int(timeLabel.split(":")[1])
            respawnSec = nowSec + deathTimer
            respawnLabel = f"{respawnSec // 60}:{respawnSec % 60:02d}"
            lines.append(f"{timeLabel} - {victimLabel} died at ({xValue}, {yValue})")
            lines.append(f"{respawnLabel} - {victimLabel} respawn")

        elif eventType == "ELITE_MONSTER_KILL":
            killer = event.get("actor")
            monster = event.get("monster")
            xValue = event.get("x", "?")
            yValue = event.get("y", "?")
            killerLabel = getChampLabel(killer, championMap)
            lines.append(f"{timeLabel} - {killerLabel} killed {monster} at ({xValue}, {yValue})")

        elif eventType == "LEVEL_UP":
            playerId = event.get("actor")
            level = event.get("level")
            levelByPlayer[str(playerId)] = level
            playerLabel = getChampLabel(playerId, championMap)
            lines.append(f"{timeLabel} - {playerLabel} leveled up to {level}")

    return lines


def buildTimelineLines(participantData: Dict[str, List[Dict]], championMap: Dict[str, Dict[str, str]]) -> List[str]:
    lines: List[str] = []
    for playerKey in sorted(participantData.keys(), key=int):
        champion = championMap[playerKey]["champion"]
        team = championMap[playerKey]["team"]
        lines.append(f"{team} {champion}")
        for row in participantData[playerKey]:
            lines.append(
                f"  {row['time']} - Position ({row['x']}, {row['y']}) | "
                f"Level {row['level']} | CS {row['cs']} | Gold {row['gold']}"
            )
        lines.append("")
    return lines


def saveText(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def processOneMatchFiles(matchPath: Path, timelinePath: Path, logDir: Path) -> bool:
    """Load pair of JSON files and write logs under logDir. Returns True if processed."""
    if not timelinePath.exists() or not matchPath.exists():
        return False

    logDir.mkdir(parents=True, exist_ok=True)

    timeline = json.loads(timelinePath.read_text(encoding="utf-8"))
    matchMeta = json.loads(matchPath.read_text(encoding="utf-8"))

    frames = timeline["info"]["frames"]
    championMap = buildChampionMap(matchMeta)
    participantData = buildTimelineData(frames)
    events = buildEventData(frames)

    parsedData = {
        "timelines": [{"id": int(playerKey), "timeline": rows} for playerKey, rows in participantData.items()],
        "events": events,
    }

    saveText(logDir / "events.log", buildEventLines(events, championMap))
    saveText(logDir / "timelines.log", buildTimelineLines(participantData, championMap))
    (logDir / "parsedData.json").write_text(json.dumps(parsedData, indent=2), encoding="utf-8")
    return True


def processDivisionJob(task: Dict) -> None:
    """All *.json matches under one division folder (paired with timelines/<run>/<division>/)."""
    runName = task["runName"]
    divisionLabel = task["divisionLabel"]
    matchesDivisionDir = Path(task["matchesDivisionDir"])
    timelinesDivisionDir = Path(task["timelinesDivisionDir"])
    dataDirPath = Path(task["dataDirStr"])

    matchFiles = sorted(matchesDivisionDir.glob("*.json"))
    print(f"[{divisionLabel}] {len(matchFiles)} match file(s) under {matchesDivisionDir.name}")

    for matchPath in matchFiles:
        matchId = matchPath.stem
        timelinePath = timelinesDivisionDir / f"{matchId}_timeline.json"
        logDir = dataDirPath / "log" / runName / divisionLabel / matchId
        if processOneMatchFiles(matchPath, timelinePath, logDir):
            print(f"[{divisionLabel}] Processed {matchId}")
        else:
            print(f"[{divisionLabel}] Skip {matchId}: missing timeline or match")


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
            print(f"[worker] failed job {label}: {exc}")
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
            print(
                f"[scheduler] dispatched {task['divisionLabel']} "
                f"(running: {sorted(inFlight)})"
            )

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
        print(f"[scheduler] finished {finished} (running: {sorted(inFlight)})")
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
    print(f"Worker processes: {numWorkers} (min of job count and CPU count)")
    print(f"Log root: {logRootDir / matchesRunRoot.name}/<division>/<matchId>/")

    runDynamicDivisionPool(tasks)
    print("\nDone.")


if __name__ == "__main__":
    main()

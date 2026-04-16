"""
Fetch match metadata and timelines for players listed in a fetchPlayer run folder.

Put one or more API keys in Data/riotApiKey.txt (one per line; # starts a comment).

Example:
  python Data/fetchMatch.py --playersFolder players/na1-ranked-solo-5x5-20260413-235836
  python Data/fetchMatch.py --playersFolder players/na1-ranked-solo-5x5-20260413-235836 --count 50

Expects that folder to contain one or more *.jsonl files (e.g. challenger-grandmaster.jsonl,
master.jsonl, diamondI.jsonl …). Each file is one job. With multiple keys, one worker process
per key runs jobs from a queue (same scheduling idea as fetchPlayer).

Writes:
  Data/matches/<run-folder-name>/<division-stem>/<matchId>.json
  Data/timelines/<run-folder-name>/<division-stem>/<matchId>_timeline.json

CLI: --playersFolder (required; run folder name starts with the platform shard, e.g. na1-...).
      --count optional; max match IDs to request per player (default 100).

Match-v5 host (americas / europe / asia) is inferred from that leading token, same grouping as
fetchPlayer uses for account routing. Optional sleeps between requests were removed; 429 backoff
in riotGet remains if Riot throttles.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import requests

dataDir = Path(__file__).resolve().parent
apiKeyPath = dataDir / "riotApiKey.txt"
playersDir = dataDir / "players"
matchesDir = dataDir / "matches"
timelinesDir = dataDir / "timelines"
TARGET_QUEUE_ID = 420  # Ranked Solo/Duo
TARGET_MAP_ID = 11     # Summoner's Rift

workerShutdown = None
PROGRESS_BAR_WIDTH = 30
PROGRESS_PRINT_EVERY_COUNT = 50
PROGRESS_PRINT_EVERY_SECONDS = 2.0


def readRiotAPIKeys() -> List[str]:
    if not apiKeyPath.exists():
        raise RuntimeError(f"Missing API key file: {apiKeyPath}")
    keys: List[str] = []
    for line in apiKeyPath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line)
    if not keys:
        raise RuntimeError(f"No API keys found in {apiKeyPath}")
    return keys


def getRiotHeaders() -> Dict[str, str]:
    key = os.environ.get("RIOT_API_KEY", "").strip()
    if not key:
        raise RuntimeError("RIOT_API_KEY is not set in the environment for this process")
    return {"X-Riot-Token": key}


def parseRetryAfterSeconds(response: requests.Response) -> Optional[float]:
    retry = response.headers.get("Retry-After")
    if retry is None:
        return None
    try:
        return float(retry)
    except ValueError:
        return None


def riotGet(url: str, headers: Dict[str, str], maxRetries: int = 12) -> requests.Response:
    attempt = 0
    while True:
        try:
            response = requests.get(url, headers=headers, timeout=60)
        except requests.exceptions.RequestException as exc:
            if attempt >= maxRetries:
                raise
            attempt += 1
            delay = min(2.0 ** min(attempt, 7), 120.0)
            print(f"[http] request error ({exc}); retrying in {delay:.1f}s")
            time.sleep(delay)
            continue

        if response.status_code == 429 and attempt < maxRetries:
            attempt += 1
            delay = parseRetryAfterSeconds(response)
            if delay is None:
                delay = min(2.0 ** min(attempt, 7), 120.0)
            time.sleep(delay)
            continue
        return response


def formatProgressBar(current: int, total: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    done = int(ratio * width)
    return f"[{'#' * done}{'-' * (width - done)}] {ratio * 100:5.1f}% ({current}/{total})"


def resolvePlayersFolder(arg: str) -> Path:
    """Resolve user path: absolute, or relative to Data, or basename under Data/players."""
    raw = Path(arg)
    if raw.is_absolute():
        resolved = raw.resolve()
        if resolved.is_dir():
            return resolved
        raise RuntimeError(f"Players folder not found: {arg}")

    candidate = (dataDir / raw).resolve()
    if candidate.is_dir():
        return candidate
    underPlayers = (playersDir / raw).resolve()
    if underPlayers.is_dir():
        return underPlayers
    raise RuntimeError(
        f"Players folder not found: {arg} (tried {candidate} and {underPlayers})"
    )


def platformRegionFromRunName(runName: str) -> str:
    """First hyphen-separated segment of fetchPlayer output dir, e.g. na1 from na1-ranked-solo-5x5-...."""
    if not runName or "-" not in runName:
        raise RuntimeError(
            f"Run folder name must start with platform-region then '-', got: {runName!r}"
        )
    token = runName.split("-", 1)[0].strip().lower()
    if not token:
        raise RuntimeError(f"Missing platform region in run folder name: {runName!r}")
    return token


def matchClusterForPlatform(platformRegion: str) -> str:
    """Match-v5 routing value (subdomain) for a platform shard: americas, europe, or asia."""
    r = platformRegion.lower().strip()
    if r in ("na1", "br1", "la1", "la2"):
        return "americas"
    if r in ("euw1", "eun1", "tr1", "ru", "me1"):
        return "europe"
    if r in ("kr", "jp1", "oc1", "ph2", "sg2", "th2", "tw2", "vn2"):
        return "asia"
    return "americas"


def loadPlayersFromJsonl(playersPath: Path) -> List[Dict]:
    if not playersPath.exists():
        raise RuntimeError(f"Missing players file: {playersPath}")

    players: List[Dict] = []
    for rawLine in playersPath.read_text(encoding="utf-8").splitlines():
        line = rawLine.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("username") and row.get("tag"):
            players.append(row)
    return players


def fetchPuuidFromRiotId(matchCluster: str, username: str, tag: str) -> str:
    encodedName = quote(username, safe="")
    encodedTag = quote(tag, safe="")
    url = (
        f"https://{matchCluster}.api.riotgames.com/riot/account/v1/accounts/"
        f"by-riot-id/{encodedName}/{encodedTag}"
    )
    response = riotGet(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(
            f"Failed to get puuid for {username}#{tag}: "
            f"{response.status_code} - {response.text}"
        )
    payload = response.json()
    puuid = payload.get("puuid")
    if not puuid:
        raise Exception(f"Missing puuid in Riot ID lookup for {username}#{tag}")
    return puuid


def fetchMatchIds(matchCluster: str, puuid: str, count: int) -> List[str]:
    url = (
        f"https://{matchCluster}.api.riotgames.com/lol/match/v5/matches/"
        f"by-puuid/{puuid}/ids?count={count}&queue={TARGET_QUEUE_ID}"
    )
    response = riotGet(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get match IDs: {response.status_code} - {response.text}")
    return response.json()


def saveJson(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def readJson(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def isEligibleMatchMetadata(matchData: Dict[str, Any]) -> bool:
    info = matchData.get("info", {})
    return info.get("queueId") == TARGET_QUEUE_ID and info.get("mapId") == TARGET_MAP_ID


def matchAlreadyStoredOrProcessed(
    dataDirPath: Path,
    runName: str,
    divisionLabel: str,
    matchId: str,
) -> bool:
    """
    Duplicate detector across raw + processed outputs.

    Checks:
      - Data/matches/<run>/<division>/<matchId>.json
      - Data/timelines/<run>/<division>/<matchId>_timeline.json
      - Data/log/<run>/<division>/<matchId>_timeline.json   (legacy naming pattern)
      - Data/log/<run>/<division>/<matchId>/parsedData.json (current processMatch output)
    """
    matchPath = dataDirPath / "matches" / runName / divisionLabel / f"{matchId}.json"
    timelinePath = (
        dataDirPath / "timelines" / runName / divisionLabel / f"{matchId}_timeline.json"
    )
    legacyLogTimelinePath = (
        dataDirPath / "log" / runName / divisionLabel / f"{matchId}_timeline.json"
    )
    processedLogPath = (
        dataDirPath / "log" / runName / divisionLabel / matchId / "parsedData.json"
    )
    return (
        matchPath.exists()
        or timelinePath.exists()
        or legacyLogTimelinePath.exists()
        or processedLogPath.exists()
    )


def fetchAndSaveMatchFiles(
    matchCluster: str,
    matchId: str,
    matchesDivisionDir: Path,
    timelinesDivisionDir: Path,
) -> None:
    timelinePath = timelinesDivisionDir / f"{matchId}_timeline.json"
    matchPath = matchesDivisionDir / f"{matchId}.json"

    matchData: Optional[Dict[str, Any]] = None
    if matchPath.exists():
        matchData = readJson(matchPath)
        if matchData is None:
            print(f"Invalid metadata JSON for {matchId}, refetching")
    if matchData is None:
        matchUrl = f"https://{matchCluster}.api.riotgames.com/lol/match/v5/matches/{matchId}"
        response = riotGet(matchUrl, headers=getRiotHeaders())
        if response.status_code == 200:
            matchData = response.json()
            saveJson(matchPath, matchData)
        else:
            print(f"Failed metadata for {matchId}: {response.status_code}")
            return

    if not isEligibleMatchMetadata(matchData):
        queueId = matchData.get("info", {}).get("queueId")
        mapId = matchData.get("info", {}).get("mapId")
        print(
            f"Skipping {matchId}: expected queue {TARGET_QUEUE_ID} on map {TARGET_MAP_ID}, "
            f"got queue {queueId}, map {mapId}"
        )
        return

    if not timelinePath.exists():
        timelineUrl = (
            f"https://{matchCluster}.api.riotgames.com/lol/match/v5/matches/"
            f"{matchId}/timeline"
        )
        response = riotGet(timelineUrl, headers=getRiotHeaders())
        if response.status_code == 200:
            saveJson(timelinePath, response.json())
        else:
            print(f"Failed timeline for {matchId}: {response.status_code}")


def runJsonlJob(task: Dict) -> None:
    """Process one division JSONL: list matches per player, save under run/division dirs."""
    dataDirPath = Path(task["dataDirStr"])
    runName = task["runName"]
    divisionLabel = task["divisionLabel"]
    jsonlPath = Path(task["jsonlPath"])
    matchCluster = task["matchCluster"]
    matchCount = int(task["matchCount"])

    matchesDivisionDir = dataDirPath / "matches" / runName / divisionLabel
    timelinesDivisionDir = dataDirPath / "timelines" / runName / divisionLabel
    matchesDivisionDir.mkdir(parents=True, exist_ok=True)
    timelinesDivisionDir.mkdir(parents=True, exist_ok=True)

    players = loadPlayersFromJsonl(jsonlPath)
    totalPlayers = len(players)
    print(f"[{divisionLabel}] {formatProgressBar(0, max(1, totalPlayers))}")

    seenRiotIds: Set[str] = set()
    seenMatchIds: Set[str] = set()
    processedPlayers = 0
    lastProgressCount = -1
    lastProgressTime = 0.0
    for player in players:
        username = player.get("username", "unknown")
        tag = player.get("tag", "unknown")
        riotIdKey = f"{str(username).strip().lower()}#{str(tag).strip().lower()}"
        if not username or not tag:
            print(f"[{divisionLabel}] Skipping entry with missing username/tag")
            continue
        if riotIdKey in seenRiotIds:
            continue
        seenRiotIds.add(riotIdKey)

        try:
            puuid = fetchPuuidFromRiotId(matchCluster, username, tag)
        except Exception as error:
            print(f"[{divisionLabel}] Failed Riot ID lookup for {username}#{tag}: {error}")
            continue

        try:
            matchIds = fetchMatchIds(matchCluster, puuid, matchCount)
        except Exception as error:
            print(f"[{divisionLabel}] Failed match list for {username}#{tag}: {error}")
        else:
            for matchId in matchIds:
                if matchId in seenMatchIds:
                    continue
                if matchAlreadyStoredOrProcessed(dataDirPath, runName, divisionLabel, matchId):
                    seenMatchIds.add(matchId)
                    continue
                seenMatchIds.add(matchId)
                fetchAndSaveMatchFiles(
                    matchCluster, matchId, matchesDivisionDir, timelinesDivisionDir
                )
        finally:
            processedPlayers += 1
            now = time.time()
            shouldPrint = (
                processedPlayers == totalPlayers
                or processedPlayers - lastProgressCount >= PROGRESS_PRINT_EVERY_COUNT
                or now - lastProgressTime >= PROGRESS_PRINT_EVERY_SECONDS
            )
            if shouldPrint:
                print(f"[{divisionLabel}] {formatProgressBar(processedPlayers, totalPlayers)}")
                lastProgressCount = processedPlayers
                lastProgressTime = now


def matchFileWorkerLoop(
    apiKey: str,
    taskQueue: "multiprocessing.Queue",
    doneQueue: "multiprocessing.Queue",
) -> None:
    while True:
        item = taskQueue.get()
        if item is workerShutdown:
            break
        task = item
        try:
            os.environ["RIOT_API_KEY"] = apiKey.strip()
            runJsonlJob(task)
            doneQueue.put(task["divisionLabel"])
        except Exception as exc:
            label = task.get("divisionLabel", "?") if isinstance(task, dict) else "?"
            print(f"[worker] failed job {label}: {exc}")
            doneQueue.put(("__error__", label, str(exc), task, apiKey.strip()))
            break


def buildTasksForFolder(
    playersFolder: Path,
    matchCluster: str,
    matchCount: int,
) -> List[Dict]:
    runName = playersFolder.name
    jsonlFiles = sorted(playersFolder.glob("*.jsonl"))
    if not jsonlFiles:
        raise RuntimeError(f"No *.jsonl files in {playersFolder}")

    dataDirStr = str(dataDir.resolve())
    tasks: List[Dict] = []
    for path in jsonlFiles:
        tasks.append(
            {
                "jsonlPath": str(path.resolve()),
                "divisionLabel": path.stem,
                "matchCluster": matchCluster,
                "matchCount": matchCount,
                "dataDirStr": dataDirStr,
                "runName": runName,
            }
        )
    return tasks


def runDynamicMatchPool(apiKeys: List[str], tasks: List[Dict]) -> None:
    numWorkers = len(apiKeys)

    if numWorkers == 1:
        os.environ["RIOT_API_KEY"] = apiKeys[0].strip()
        for task in tasks:
            runJsonlJob(task)
        return

    manager = multiprocessing.Manager()
    taskQueue = manager.Queue()
    doneQueue = manager.Queue()

    workers: List[multiprocessing.Process] = []
    for key in apiKeys:
        proc = multiprocessing.Process(
            target=matchFileWorkerLoop,
            args=(key, taskQueue, doneQueue),
        )
        proc.start()
        workers.append(proc)

    inFlight: Set[str] = set()
    pendingTasks = deque(tasks)
    activeWorkers = numWorkers
    failedJobs: List[str] = []
    retiredKeys: List[str] = []

    def tryDispatch() -> None:
        while len(inFlight) < activeWorkers and pendingTasks:
            task = pendingTasks.popleft()
            taskQueue.put(task)
            inFlight.add(task["divisionLabel"])

    tryDispatch()

    while inFlight or pendingTasks:
        if not inFlight:
            if not pendingTasks:
                break
            if activeWorkers <= 0:
                raise RuntimeError(
                    "All API keys failed; no workers left to continue pending jobs."
                )
            tryDispatch()
            if not inFlight:
                continue

        finished = doneQueue.get()
        if isinstance(finished, tuple) and finished[0] == "__error__":
            failedLabel = finished[1]
            failedError = finished[2]
            failedTask = finished[3]
            failedKey = finished[4]
            inFlight.discard(failedLabel)
            failedJobs.append(f"{failedLabel}: {failedError}")
            pendingTasks.append(failedTask)
            retiredKeys.append(failedKey)
            activeWorkers = max(0, activeWorkers - 1)
            print(
                f"[scheduler] failed {failedLabel} (running: {sorted(inFlight)}) "
                f"-> key retired ({failedKey}), job requeued"
            )
            tryDispatch()
            continue
        inFlight.discard(finished)
        tryDispatch()
    for _ in workers:
        taskQueue.put(workerShutdown)
    for proc in workers:
        proc.join(timeout=600)
        if proc.is_alive():
            proc.terminate()

    if failedJobs:
        print("\nCompleted with failed jobs:")
        for row in failedJobs:
            print(f"  - {row}")
    if retiredKeys:
        print("\nRetired API keys:")
        for key in retiredKeys:
            print(f"  - {key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch matches/timelines for each JSONL in a fetchPlayer run folder."
    )
    parser.add_argument(
        "--playersFolder",
        required=True,
        help="Run folder under Data/players, e.g. players/na1-ranked-solo-5x5-20260413-235836",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Max match IDs to request per player",
    )
    args = parser.parse_args()

    apiKeys = readRiotAPIKeys()
    print(f"Loaded {len(apiKeys)} API key(s) from {apiKeyPath}")

    playersFolder = resolvePlayersFolder(args.playersFolder)
    runName = playersFolder.name
    platformRegion = platformRegionFromRunName(runName)
    matchCluster = matchClusterForPlatform(platformRegion)
    tasks = buildTasksForFolder(playersFolder, matchCluster, args.count)
    print(f"Players folder: {playersFolder}")
    print(f"Platform {platformRegion!r} -> match cluster {matchCluster!r}")
    print(f"Jobs ({len(tasks)}): " + " → ".join(t["divisionLabel"] for t in tasks))
    print(f"Output: {matchesDir / runName}/<division>/ and {timelinesDir / runName}/<division>/")

    runDynamicMatchPool(apiKeys, tasks)
    print("\nDone.")


if __name__ == "__main__":
    main()

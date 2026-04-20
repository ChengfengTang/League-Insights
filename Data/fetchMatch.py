"""
Fetch match metadata and timelines for players listed in a fetchPlayer run folder.

Put one or more API keys in Data/riotApiKey.txt (one per line; # starts a comment).

Example:
  python3 Data/fetchMatch.py --playersFolder players/na1-ranked-solo-5x5-20260413-235836
  python3 Data/fetchMatch.py --playersFolder players/na1-ranked-solo-5x5-20260413-235836 --count 50

Expects that folder to contain one or more *.jsonl files (e.g. challenger-grandmaster.jsonl,
master.jsonl, diamondI.jsonl …). All players are merged and split evenly across workers (one
worker per API key). Match files are written flat under the run folder (no division subfolders).

Writes:
  Data/matches/<run-folder-name>/<matchId>.json
  Data/timelines/<run-folder-name>/<matchId>_timeline.json

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
PROGRESS_PRINT_EVERY_COUNT = 100
PROGRESS_PRINT_EVERY_SECONDS = 10.0


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
            print(f"[http] request error ({exc}); retrying in {delay:.1f}s", flush=True)
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


def matchAlreadyStoredOrProcessed(dataDirPath: Path, runName: str, matchId: str) -> bool:
    """
    Duplicate detector across raw + processed outputs for this run.

    Checks:
      - Data/matches/<run>/<matchId>.json
      - Data/timelines/<run>/<matchId>_timeline.json
      - Data/log/<run>/.done/<matchId>
      - Data/log/<run>/<ChampionName>/<matchId>_p<participantId>_<side>.json
    """
    matchPath = dataDirPath / "matches" / runName / f"{matchId}.json"
    timelinePath = dataDirPath / "timelines" / runName / f"{matchId}_timeline.json"
    logRunDir = dataDirPath / "log" / runName
    doneMarkerPath = logRunDir / ".done" / matchId
    groupedLogExists = any(logRunDir.glob(f"*/{matchId}_p*_*.json"))
    return (
        matchPath.exists()
        or timelinePath.exists()
        or doneMarkerPath.exists()
        or groupedLogExists
    )


def fetchAndSaveMatchFiles(
    matchCluster: str,
    matchId: str,
    matchesRunDir: Path,
    timelinesRunDir: Path,
) -> None:
    matchesRunDir.mkdir(parents=True, exist_ok=True)
    timelinesRunDir.mkdir(parents=True, exist_ok=True)
    timelinePath = timelinesRunDir / f"{matchId}_timeline.json"
    matchPath = matchesRunDir / f"{matchId}.json"

    matchData: Optional[Dict[str, Any]] = None
    if matchPath.exists():
        matchData = readJson(matchPath)
        if matchData is None:
            print(f"Invalid metadata JSON for {matchId}, refetching", flush=True)
    if matchData is None:
        matchUrl = f"https://{matchCluster}.api.riotgames.com/lol/match/v5/matches/{matchId}"
        response = riotGet(matchUrl, headers=getRiotHeaders())
        if response.status_code == 200:
            matchData = response.json()
            saveJson(matchPath, matchData)
        else:
            print(f"Failed metadata for {matchId}: {response.status_code}", flush=True)
            return

    if not isEligibleMatchMetadata(matchData):
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
            print(f"Failed timeline for {matchId}: {response.status_code}", flush=True)


def loadAllPlayersFromRunFolder(playersFolder: Path) -> List[Dict]:
    """Load every *.jsonl in the run folder; dedupe by riotId (username#tag) case-insensitive."""
    seenKeys: Set[str] = set()
    merged: List[Dict] = []
    for jsonlPath in sorted(playersFolder.glob("*.jsonl")):
        for row in loadPlayersFromJsonl(jsonlPath):
            username = row.get("username", "")
            tag = row.get("tag", "")
            riotIdKey = f"{str(username).strip().lower()}#{str(tag).strip().lower()}"
            if riotIdKey in seenKeys:
                continue
            seenKeys.add(riotIdKey)
            merged.append(row)
    return merged


def splitPlayersIntoShards(players: List[Dict], numShards: int) -> List[List[Dict]]:
    if numShards <= 0:
        return [players]
    numShards = min(numShards, max(1, len(players)))
    shards: List[List[Dict]] = [[] for _ in range(numShards)]
    for index, player in enumerate(players):
        shards[index % numShards].append(player)
    return [s for s in shards if s]


def runPlayerShardJob(task: Dict) -> None:
    """Fetch match lists for a shard of players; write under matches/<run>/ and timelines/<run>/."""
    dataDirPath = Path(task["dataDirStr"])
    runName = task["runName"]
    shardLabel = task["shardLabel"]
    players: List[Dict] = task["players"]
    matchCluster = task["matchCluster"]
    matchCount = int(task["matchCount"])

    matchesRunDir = dataDirPath / "matches" / runName
    timelinesRunDir = dataDirPath / "timelines" / runName

    totalPlayers = len(players)
    print(f"[{shardLabel}] {formatProgressBar(0, max(1, totalPlayers))}", flush=True)

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
            print(f"[{shardLabel}] Skipping entry with missing username/tag", flush=True)
            continue
        if riotIdKey in seenRiotIds:
            continue
        seenRiotIds.add(riotIdKey)

        try:
            puuid = fetchPuuidFromRiotId(matchCluster, username, tag)
        except Exception as error:
            print(f"[{shardLabel}] Failed Riot ID lookup for {username}#{tag}: {error}", flush=True)
            continue

        try:
            matchIds = fetchMatchIds(matchCluster, puuid, matchCount)
        except Exception as error:
            print(f"[{shardLabel}] Failed match list for {username}#{tag}: {error}", flush=True)
        else:
            for matchId in matchIds:
                if matchId in seenMatchIds:
                    continue
                if matchAlreadyStoredOrProcessed(dataDirPath, runName, matchId):
                    seenMatchIds.add(matchId)
                    continue
                seenMatchIds.add(matchId)
                fetchAndSaveMatchFiles(matchCluster, matchId, matchesRunDir, timelinesRunDir)
        finally:
            processedPlayers += 1
            now = time.time()
            shouldPrint = (
                processedPlayers == totalPlayers
                or processedPlayers - lastProgressCount >= PROGRESS_PRINT_EVERY_COUNT
                or now - lastProgressTime >= PROGRESS_PRINT_EVERY_SECONDS
            )
            if shouldPrint:
                print(
                    f"[{shardLabel}] {formatProgressBar(processedPlayers, totalPlayers)}",
                    flush=True,
                )
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
            runPlayerShardJob(task)
            doneQueue.put(task["shardLabel"])
        except Exception as exc:
            label = task.get("shardLabel", "?") if isinstance(task, dict) else "?"
            print(f"[worker] failed job {label}: {exc}", flush=True)
            doneQueue.put(("__error__", label, str(exc), task, apiKey.strip()))
            break


def buildShardTasksForFolder(
    playersFolder: Path,
    matchCluster: str,
    matchCount: int,
    numShards: int,
) -> List[Dict]:
    runName = playersFolder.name
    if not any(playersFolder.glob("*.jsonl")):
        raise RuntimeError(f"No *.jsonl files in {playersFolder}")

    allPlayers = loadAllPlayersFromRunFolder(playersFolder)
    if not allPlayers:
        raise RuntimeError(f"No valid player rows in *.jsonl under {playersFolder}")

    dataDirStr = str(dataDir.resolve())
    shards = splitPlayersIntoShards(allPlayers, numShards)
    tasks: List[Dict] = []
    for index, shard in enumerate(shards):
        tasks.append(
            {
                "players": shard,
                "shardLabel": f"shard-{index}",
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
            runPlayerShardJob(task)
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
            inFlight.add(task["shardLabel"])

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
            , flush=True)
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
        print("\nCompleted with failed jobs:", flush=True)
        for row in failedJobs:
            print(f"  - {row}", flush=True)
    if retiredKeys:
        print("\nRetired API keys:", flush=True)
        for key in retiredKeys:
            print(f"  - {key}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch matches/timelines for all players in a fetchPlayer run (merged, sharded by API key)."
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
    print(f"Loaded {len(apiKeys)} API key(s) from {apiKeyPath}", flush=True)

    playersFolder = resolvePlayersFolder(args.playersFolder)
    runName = playersFolder.name
    platformRegion = platformRegionFromRunName(runName)
    matchCluster = matchClusterForPlatform(platformRegion)
    tasks = buildShardTasksForFolder(playersFolder, matchCluster, args.count, len(apiKeys))
    print(f"Players folder: {playersFolder}", flush=True)
    print(f"Platform {platformRegion!r} -> match cluster {matchCluster!r}", flush=True)
    totalPlayers = sum(len(t["players"]) for t in tasks)
    print(
        f"Shards ({len(tasks)}) for {len(apiKeys)} key(s); {totalPlayers} unique players",
        flush=True,
    )
    print(f"Output: {matchesDir / runName}/ and {timelinesDir / runName}/", flush=True)

    runDynamicMatchPool(apiKeys, tasks)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

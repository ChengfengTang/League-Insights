"""
Fetch ranked players (challenger through diamond IV) from Riot and write JSONL files.

Put API keys in Data/riotApiKey.txt (one per line; lines starting with # are ignored).
Example: python Data/fetchPlayer.py --region na1 --queue RANKED_SOLO_5x5

For each run, data goes under Data/players/<region>-<queue>-<timestamp>/ as flat files:
challenger-grandmaster.jsonl, master.jsonl, diamondI.jsonl through diamondIV.jsonl.
Each line is JSON: username, tag, puuid, sourceTier.

Work is split into six jobs (see tierJobsSpec). With one key, jobs run in order in this
process. With several keys, one worker process is started per key and a small scheduler keeps
up to that many jobs running; each job writes its own filename so outputs never collide.

CLI: --region (e.g. na1, euw1, kr), --queue (e.g. RANKED_SOLO_5x5, RANKED_FLEX_SR).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# This file lives in Data/; paths below are relative to that folder.
dataDir = Path(__file__).resolve().parent
apiKeyPath = dataDir / "riotApiKey.txt"
playersDir = dataDir / "players"

# Dispatch order 0..N-1. Tuple: log label, jobKind string, API tier string, diamond division or "".
# jobKind values: challenger_grandmaster (challenger + GM leagues), apex (master league),
#                 diamond_division (paginated DIAMOND + division).
tierJobsSpec: List[Tuple[str, str, str, str]] = [
    ("challenger-grandmaster", "challenger_grandmaster", "", ""),
    ("master", "apex", "master", ""),
    ("diamond-I", "diamond_division", "DIAMOND", "I"),
    ("diamond-II", "diamond_division", "DIAMOND", "II"),
    ("diamond-III", "diamond_division", "DIAMOND", "III"),
    ("diamond-IV", "diamond_division", "DIAMOND", "IV"),
]
numTierJobs = len(tierJobsSpec)

workerShutdown = None  # task-queue sentinel so child processes exit their loops.


@dataclass
class TierJob:
    """
    One unit of work sent to a worker process (must be picklable / JSON-friendly).

    Fields:
        outputFolder: Label for logs / scheduling (e.g. "diamond-II"); not a path.
        jobKind:      Which branch in materializeTierEntries() runs.
        tierUpper:    Riot API tier string when needed ("master", "DIAMOND").
        division:     For diamond only: "I", "II", "III", or "IV".
    """

    outputFolder: str
    jobKind: str
    tierUpper: str = ""
    division: str = ""


def readRiotAPIKeys() -> List[str]:
    """Load all non-empty, non-comment lines from riotApiKey.txt."""
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
    """
    Authorization header for the current OS process.

    Child workers set os.environ["RIOT_API_KEY"] before calling this, so each
    process can use a different development key (separate rate-limit buckets).
    """
    key = os.environ.get("RIOT_API_KEY", "").strip()
    if not key:
        raise RuntimeError("RIOT_API_KEY is not set in the environment for this process")
    return {"X-Riot-Token": key}


def parseRetryAfterSeconds(response: requests.Response) -> Optional[float]:
    """If Riot sends Retry-After on 429, return seconds to sleep (float)."""
    retry = response.headers.get("Retry-After")
    if retry is None:
        return None
    try:
        return float(retry)
    except ValueError:
        return None


def riotGet(url: str, headers: Dict[str, str], maxRetries: int = 12) -> requests.Response:
    """
    GET with automatic retry on HTTP 429.

    Uses Retry-After header when present; otherwise exponential backoff capped
    at 120 seconds. Non-429 responses are returned immediately (including 4xx/5xx).
    """
    attempt = 0
    while True:
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code == 429 and attempt < maxRetries:
            attempt += 1
            delay = parseRetryAfterSeconds(response)
            if delay is None:
                delay = min(2.0 ** min(attempt, 7), 120.0)
            time.sleep(delay)
            continue
        return response


def fetchApexTierPlayers(region: str, tier: str, queue: str) -> List[Dict]:
    """
    One apex league snapshot: challenger, grandmaster, or master.

    Endpoint shape: .../lol/league/v4/{tier}leagues/by-queue/{queue}
    Returns the "entries" list sorted by league points (high → low).
    """
    url = f"https://{region}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/{queue}"
    response = riotGet(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get {tier} players: {response.status_code} - {response.text}")
    entries = response.json().get("entries", [])
    entries.sort(key=lambda entry: entry.get("leaguePoints", 0), reverse=True)
    return entries


def fetchChallengerGrandmasterEntries(region: str, queue: str) -> List[Dict]:
    """
    Combined job: challenger entries first, then grandmaster (single output file).

    sourceTier is set on each row so JSONL consumers know which apex bucket
    the player came from.
    """
    out: List[Dict] = []
    for tier in ["challenger", "grandmaster"]:
        entries = fetchApexTierPlayers(region=region, tier=tier, queue=queue)
        for entry in entries:
            entry["sourceTier"] = tier
        out.extend(entries)
    return out


def fetchDiamondDivisionPages(
    region: str,
    queue: str,
    division: str,
    logPrefix: str,
) -> List[Dict]:
    """
    Walk every page of league/v4/entries for DIAMOND + one division (I–IV).

    Stops when a page returns an empty list (no more players in that division).
    Typical page size ~200; large regions need many pages per division.
    """
    tier = "DIAMOND"
    allEntries: List[Dict] = []
    page = 1
    while True:
        url = (
            f"https://{region}.api.riotgames.com/lol/league/v4/entries/"
            f"{queue}/{tier}/{division}?page={page}"
        )
        response = riotGet(url, headers=getRiotHeaders())
        if response.status_code != 200:
            raise Exception(
                f"Failed to get {tier} {division} page {page}: "
                f"{response.status_code} - {response.text}"
            )
        entries = response.json()
        if not entries:
            break
        entries.sort(key=lambda entry: entry.get("leaguePoints", 0), reverse=True)
        allEntries.extend(entries)
        print(
            f"[{logPrefix}] League fetch {tier} div {division} page {page}: "
            f"+{len(entries)} rows (total {len(allEntries)})"
        )
        page += 1
    return allEntries


def accountApiHost(platformRegion: str) -> str:
    """
    Regional hostname for /riot/account/v1 (NOT the same as platform routing).

    By-puuid lookup must hit the cluster that owns that shard (e.g. kr → asia).
    """
    r = platformRegion.lower().strip()
    if r in ("na1", "br1", "la1", "la2"):
        return "americas.api.riotgames.com"
    if r in ("euw1", "eun1", "tr1", "ru", "me1"):
        return "europe.api.riotgames.com"
    if r in ("kr", "jp1", "oc1", "ph2", "sg2", "th2", "tw2", "vn2"):
        return "asia.api.riotgames.com"
    return "americas.api.riotgames.com"


def getRiotId(platformRegion: str, puuid: str) -> Dict:
    """Riot Account API: puuid → { gameName, tagLine, ... }."""
    host = accountApiHost(platformRegion)
    url = f"https://{host}/riot/account/v1/accounts/by-puuid/{puuid}"
    response = riotGet(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get Riot ID: {response.status_code} - {response.text}")
    return response.json()


def getPuuidFromSummonerId(region: str, summonerId: str) -> str:
    """Fallback when a league row has summonerId but no puuid yet."""
    url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/{summonerId}"
    response = riotGet(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get summoner info: {response.status_code} - {response.text}")
    return response.json()["puuid"]


def sanitizeToken(value: str) -> str:
    """Make a string safe for use in a directory name (lowercase, alnum + hyphen)."""
    cleanedChars = []
    for char in value.lower():
        if char.isalnum():
            cleanedChars.append(char)
        else:
            cleanedChars.append("-")
    cleaned = "".join(cleanedChars).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "unknown"


def buildRunDir(region: str, queue: str) -> Path:
    """Create Data/players/<region>-<queue>-<timestamp>/ and return its Path."""
    playersDir.mkdir(parents=True, exist_ok=True)
    regionToken = sanitizeToken(region)
    queueToken = sanitizeToken(queue)
    timestampToken = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirName = f"{regionToken}-{queueToken}-{timestampToken}"
    runDir = playersDir / dirName
    runDir.mkdir(parents=True, exist_ok=True)
    return runDir


def jobJsonlPath(runDir: Path, job: TierJob) -> Path:
    """Flat JSONL path under runDir: e.g. diamondII.jsonl, master.jsonl (no subfolders)."""
    if job.jobKind == "challenger_grandmaster":
        return runDir / "challenger-grandmaster.jsonl"
    if job.jobKind == "apex":
        return runDir / f"{job.tierUpper.lower()}.jsonl"
    if job.jobKind == "diamond_division":
        if job.division not in ("I", "II", "III", "IV"):
            raise ValueError(f"Unknown diamond division: {job.division!r}")
        return runDir / f"diamond{job.division}.jsonl"
    raise ValueError(f"Unknown jobKind: {job.jobKind}")


def resolveAndWriteEntries(
    entries: List[Dict],
    region: str,
    outPath: Path,
    workerLabel: str,
) -> None:
    """
    For each league entry: ensure puuid, fetch Riot ID, append one JSON line.

    Skips duplicate puuids within this file only. Logs per-player lines and
    continues on individual row errors so one bad row does not abort the job.
    """
    outPath.parent.mkdir(parents=True, exist_ok=True)
    seenPuuids: Set[str] = set()
    with outPath.open("w", encoding="utf-8") as outFile:
        for entry in entries:
            try:
                puuid = entry.get("puuid")
                if not puuid:
                    summonerId = entry.get("summonerId")
                    if not summonerId:
                        raise KeyError("Missing both 'puuid' and 'summonerId' in league entry")
                    puuid = getPuuidFromSummonerId(region, summonerId)

                if puuid in seenPuuids:
                    continue

                riotId = getRiotId(region, puuid)

                playerRow = {
                    "username": riotId["gameName"],
                    "tag": riotId["tagLine"],
                    "puuid": puuid,
                    "sourceTier": entry.get("sourceTier", "unknown"),
                }
                outFile.write(json.dumps(playerRow, ensure_ascii=False) + "\n")
                seenPuuids.add(puuid)

                print(
                    f"[{workerLabel}] {riotId['gameName']}#{riotId['tagLine']} "
                    f"(tier={playerRow['sourceTier']})"
                )
            except Exception as error:
                summonerLabel = (
                    entry.get("summonerName")
                    or entry.get("summonerId")
                    or entry.get("puuid")
                    or "<unknown-entry>"
                )
                print(f"[{workerLabel}] Error processing {summonerLabel}: {error}")
                continue


def materializeTierEntries(job: TierJob, region: str, queue: str) -> List[Dict]:
    """Dispatch on jobKind: return the full list of league dicts for this job."""
    if job.jobKind == "challenger_grandmaster":
        return fetchChallengerGrandmasterEntries(region, queue)
    if job.jobKind == "apex":
        entries = fetchApexTierPlayers(region, job.tierUpper, queue)
        for entry in entries:
            entry["sourceTier"] = job.tierUpper
        return entries
    if job.jobKind == "diamond_division":
        entries = fetchDiamondDivisionPages(region, queue, job.division, job.outputFolder)
        tierLower = job.tierUpper.lower()
        for entry in entries:
            entry["sourceTier"] = tierLower
        return entries
    raise ValueError(f"Unknown jobKind: {job.jobKind}")


def runTierJob(job: TierJob, region: str, queue: str, runDir: Path) -> None:
    """Fetch one league slice and write resolved rows to a flat file under runDir."""
    print(f"[{job.outputFolder}] Fetching league entries...")
    entries = materializeTierEntries(job, region, queue)
    outPath = jobJsonlPath(runDir, job)
    print(f"[{job.outputFolder}] Resolving {len(entries)} entries -> {outPath}")
    resolveAndWriteEntries(entries, region, outPath, job.outputFolder)


def tierWorkerLoop(apiKey: str, taskQueue: "multiprocessing.Queue", doneQueue: "multiprocessing.Queue") -> None:
    """
    Child process entry point: block on taskQueue, run jobs until shutdown sentinel.

    Each task is a tuple (jobDict, runDirStr, region, queue). On success,
    pushes outputFolder name to doneQueue so the parent can track in-flight jobs.
    On failure, pushes ("__error__", message) and the parent aborts the pool.
    """
    while True:
        item = taskQueue.get()
        if item is workerShutdown:
            break
        jobDict, runDirStr, region, queue = item
        try:
            os.environ["RIOT_API_KEY"] = apiKey.strip()
            job = TierJob(**jobDict)
            runTierJob(job, region, queue, Path(runDirStr))
            doneQueue.put(job.outputFolder)
        except Exception as exc:
            print(f"[worker] failed job {jobDict.get('outputFolder', '?')}: {exc}")
            doneQueue.put(("__error__", str(exc)))


def buildJobAtIndex(jobIndex: int) -> TierJob:
    """Map a row in tierJobsSpec to a TierJob instance."""
    folder, kind, tierUpper, division = tierJobsSpec[jobIndex]
    if kind == "challenger_grandmaster":
        return TierJob(folder, kind)
    if kind == "apex":
        return TierJob(folder, kind, tierUpper)
    if kind == "diamond_division":
        return TierJob(folder, kind, tierUpper, division)
    raise ValueError(f"Unknown job kind: {kind}")


def runDynamicTierPool(
    apiKeys: List[str],
    runDir: Path,
    region: str,
    queue: str,
) -> None:
    """
    Run all numTierJobs jobs with up to len(apiKeys) concurrent workers.

    Single-key mode avoids multiprocessing overhead: same code path, sequential.

    Multi-key mode:
      - Spawn one Process per key; each runs tierWorkerLoop with that key.
      - Parent maintains nextJobIndex (jobs 0..N-1 in order) and inFlight set.
      - tryDispatch() fills empty worker slots until no jobs left or all workers busy.
      - Main loop waits on doneQueue; when a job completes, remove from inFlight
        and tryDispatch again until every job has been started and finished.
    """
    runDirResolved = str(runDir.resolve())
    numWorkers = len(apiKeys)

    if numWorkers == 1:
        os.environ["RIOT_API_KEY"] = apiKeys[0].strip()
        for idx in range(numTierJobs):
            runTierJob(buildJobAtIndex(idx), region, queue, runDir)
        return

    manager = multiprocessing.Manager()
    taskQueue = manager.Queue()
    doneQueue = manager.Queue()

    workers: List[multiprocessing.Process] = []
    for key in apiKeys:
        proc = multiprocessing.Process(
            target=tierWorkerLoop,
            args=(key, taskQueue, doneQueue),
        )
        proc.start()
        workers.append(proc)

    inFlight: Set[str] = set()  # job labels still running on a worker
    nextJobIndex = 0  # next slot in tierJobsSpec to hand out

    def tryDispatch() -> None:
        """Start as many pending jobs as we have idle workers and remaining work."""
        nonlocal nextJobIndex
        while len(inFlight) < numWorkers and nextJobIndex < numTierJobs:
            job = buildJobAtIndex(nextJobIndex)
            nextJobIndex += 1
            taskQueue.put((dataclasses.asdict(job), runDirResolved, region, queue))
            inFlight.add(job.outputFolder)
            print(
                f"[scheduler] dispatched {job.outputFolder} "
                f"(running: {sorted(inFlight)})"
            )

    tryDispatch()

    while inFlight or nextJobIndex < numTierJobs:
        if not inFlight:
            if nextJobIndex >= numTierJobs:
                break
            tryDispatch()
            if not inFlight:
                continue

        folder = doneQueue.get()
        if isinstance(folder, tuple) and folder[0] == "__error__":
            for _ in workers:
                taskQueue.put(workerShutdown)
            raise RuntimeError(folder[1])
        inFlight.discard(folder)
        print(f"[scheduler] finished {folder} (running: {sorted(inFlight)})")

        tryDispatch()

    for _ in workers:
        taskQueue.put(workerShutdown)
    for proc in workers:
        proc.join(timeout=600)
        if proc.is_alive():
            proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch challenger through diamond IV; one flat JSONL per tier/division under the run dir."
    )
    parser.add_argument("--region", default="na1", help="Regional routing (na1, euw1, kr, etc.)")
    parser.add_argument("--queue", default="RANKED_SOLO_5x5", help="Queue type")
    args = parser.parse_args()

    apiKeys = readRiotAPIKeys()
    print(f"Loaded {len(apiKeys)} API key(s) from {apiKeyPath}")

    runDir = buildRunDir(args.region, args.queue)
    print(f"Run directory: {runDir}")
    outNames = [jobJsonlPath(runDir, buildJobAtIndex(i)).name for i in range(numTierJobs)]
    print(f"Output files under {runDir}: {' → '.join(outNames)}")

    runDynamicTierPool(apiKeys, runDir, args.region, args.queue)
    print("\nDone.")


if __name__ == "__main__":
    main()

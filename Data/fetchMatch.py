"""
Fetch match metadata + timeline files for players from a selected Data/players JSONL file.

How to run:
- Put Riot API key in Data/riotApiKey.txt
- Ensure a players snapshot exists in Data/players/ (from fetchPlayer.py)
- Run: python Data/fetchMatch.py --playersFile top300-ranked-solo-5x5-na1-20260413-175452.jsonl --count 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set

import requests


dataDir = Path(__file__).resolve().parent
apiKeyPath = dataDir / "riotApiKey.txt"
playersDir = dataDir / "players"
matchesDir = dataDir / "matches"
timelinesDir = dataDir / "timelines"


def readRiotApiKey() -> str:
    """Read Riot API key from local file."""
    if not apiKeyPath.exists():
        raise RuntimeError(f"Missing API key file: {apiKeyPath}")
    apiKey = apiKeyPath.read_text(encoding="utf-8").strip()
    if not apiKey:
        raise RuntimeError(f"API key file is empty: {apiKeyPath}")
    return apiKey


riotApiKey = readRiotApiKey()


def getRiotHeaders() -> Dict[str, str]:
    """Build Riot auth headers."""
    return {"X-Riot-Token": riotApiKey}


def loadPlayersFromJsonl(playersPath: Path) -> List[Dict]:
    """
    Load player rows from Data/topRankedPlayers.jsonl.
    Expected fields include at least: username, tag, puuid.
    """
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
        if row.get("puuid"):
            players.append(row)
    return players


def fetchMatchIds(matchRegion: str, puuid: str, count: int) -> List[str]:
    """Fetch recent match IDs for one player."""
    url = (
        f"https://{matchRegion}.api.riotgames.com/lol/match/v5/matches/"
        f"by-puuid/{puuid}/ids?count={count}"
    )
    response = requests.get(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get match IDs: {response.status_code} - {response.text}")
    return response.json()


def saveJson(path: Path, payload: Dict) -> None:
    """Write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetchAndSaveMatchFiles(matchRegion: str, matchId: str) -> None:
    """Download and save timeline + metadata JSON for one match."""
    timelinePath = timelinesDir / f"{matchId}_timeline.json"
    matchPath = matchesDir / f"{matchId}.json"

    if not timelinePath.exists():
        timelineUrl = f"https://{matchRegion}.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline"
        response = requests.get(timelineUrl, headers=getRiotHeaders())
        if response.status_code == 200:
            saveJson(timelinePath, response.json())
            print(f"Saved timeline for {matchId}")
        else:
            print(f"Failed timeline for {matchId}: {response.status_code}")
    else:
        print(f"Timeline exists for {matchId}, skipping")

    if not matchPath.exists():
        matchUrl = f"https://{matchRegion}.api.riotgames.com/lol/match/v5/matches/{matchId}"
        response = requests.get(matchUrl, headers=getRiotHeaders())
        if response.status_code == 200:
            saveJson(matchPath, response.json())
            print(f"Saved metadata for {matchId}")
        else:
            print(f"Failed metadata for {matchId}: {response.status_code}")
    else:
        print(f"Metadata exists for {matchId}, skipping")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch match/timeline data for top-ranked players.")
    parser.add_argument(
        "--playersFile",
        required=True,
        help="File name under Data/players, e.g. top300-ranked-solo-5x5-na1-20260413-175452.jsonl",
    )
    parser.add_argument("--count", type=int, default=100, help="Matches per player to request")
    parser.add_argument("--matchRegion", default="americas", help="Match-v5 region, e.g. americas/europe/asia")
    parser.add_argument("--sleepSec", type=float, default=1.0, help="Delay between API requests")
    args = parser.parse_args()

    playersPath = playersDir / Path(args.playersFile).name
    matchesDir.mkdir(parents=True, exist_ok=True)
    timelinesDir.mkdir(parents=True, exist_ok=True)

    players = loadPlayersFromJsonl(playersPath)
    print(f"Loaded {len(players)} players from {playersPath}")

    seenMatchIds: Set[str] = set()
    for player in players:
        username = player.get("username", "unknown")
        tag = player.get("tag", "unknown")
        puuid = player.get("puuid")
        if not puuid:
            print(f"Skipping {username}#{tag}: missing puuid")
            continue

        try:
            matchIds = fetchMatchIds(args.matchRegion, puuid, args.count)
            print(f"Found {len(matchIds)} matches for {username}#{tag}")
        except Exception as error:
            print(f"Failed match list for {username}#{tag}: {error}")
            continue

        time.sleep(args.sleepSec)

        for matchId in matchIds:
            if matchId in seenMatchIds:
                continue
            seenMatchIds.add(matchId)
            fetchAndSaveMatchFiles(args.matchRegion, matchId)
            time.sleep(args.sleepSec)


if __name__ == "__main__":
    main()

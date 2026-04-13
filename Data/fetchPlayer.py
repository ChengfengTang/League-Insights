"""
Fetch top-ranked League players and stream-write them to local JSONL.

How to run:
- Put your Riot key in Data/riotApiKey.txt (single line).
- From repo root:
  python Data/fetchPlayer.py -n 300 --region na1 --queue RANKED_SOLO_5x5

What it does:
- Fetches ranked players from highest to lowest tiers until n players are collected.
- Resolves each player to Riot ID (gameName, tagLine) + puuid.
- Writes one JSON object per line to a new per-run file in Data/players/.

Parameters:
- -n, --count: number of players to fetch (default 300)
- -region: routing region (e.g. na1, euw1, kr)
- -queue: queue type (e.g. RANKED_SOLO_5x5, RANKED_FLEX_SR)
"""

import argparse
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

dataDir = Path(__file__).resolve().parent
apiKeyPath = dataDir / "riotApiKey.txt"
playersDir = dataDir / "players"


def readRiotApiKey() -> str:
    """Read Riot API key from `Data/riot_api_key.txt`."""
    if not apiKeyPath.exists():
        raise RuntimeError(f"Missing API key file: {apiKeyPath}")
    apiKey = apiKeyPath.read_text(encoding="utf-8").strip()
    if not apiKey:
        raise RuntimeError(f"API key file is empty: {apiKeyPath}")
    return apiKey

riotApiKey = readRiotApiKey()


def getRiotHeaders() -> Dict[str, str]:
    """Build Riot API headers."""
    return {"X-Riot-Token": riotApiKey}


def fetchApexTierPlayers(region: str, tier: str, queue: str) -> List[Dict]:
    """Fetch Challenger/Grandmaster/Master entries."""
    url = f"https://{region}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/{queue}"
    response = requests.get(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get {tier} players: {response.status_code} - {response.text}")
    entries = response.json().get("entries", [])
    entries.sort(key=lambda entry: entry.get("leaguePoints", 0), reverse=True)
    return entries


def fetchRegularTierPlayers(region: str, tier: str, queue: str, maxCount: int) -> List[Dict]:
    """Fetch Diamond and below from divisions/pages, stopping at maxCount."""
    allEntries: List[Dict] = []
    for division in ["I", "II", "III", "IV"]:
        if len(allEntries) >= maxCount:
            break
        page = 1
        while True:
            if len(allEntries) >= maxCount:
                break
            url = (
                f"https://{region}.api.riotgames.com/lol/league/v4/entries/"
                f"{queue}/{tier}/{division}?page={page}"
            )
            response = requests.get(url, headers=getRiotHeaders())
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
            page += 1
    return allEntries[:maxCount]


def getPlayers(n: int, region: str = "na1", queue: str = "RANKED_SOLO_5x5") -> List[Dict]:
    """Fetch top N players by walking tiers from high to low."""
    if n <= 0:
        return []

    tierOrder = [
        "challenger",
        "grandmaster",
        "master",
        "diamond",
        "emerald",
        "platinum",
        "gold",
        "silver",
        "bronze",
        "iron",
    ]
    apexTiers = {"challenger", "grandmaster", "master"}
    selectedEntries: List[Dict] = []

    for tier in tierOrder:
        if len(selectedEntries) >= n:
            break
        print(f"Fetching {tier} players...")
        if tier in apexTiers:
            entries = fetchApexTierPlayers(region=region, tier=tier, queue=queue)
        else:
            needed = n - len(selectedEntries)
            entries = fetchRegularTierPlayers(
                region=region,
                tier=tier.upper(),
                queue=queue,
                maxCount=needed,
            )

        for entry in entries:
            entry["sourceTier"] = tier
            selectedEntries.append(entry)
            if len(selectedEntries) >= n:
                break

    return selectedEntries


def getRiotId(puuid: str) -> Dict:
    """Resolve `puuid` to Riot ID (`gameName`, `tagLine`)."""
    url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
    response = requests.get(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get Riot ID: {response.status_code} - {response.text}")
    return response.json()


def getPuuidFromSummonerId(region: str, summonerId: str) -> str:
    """Resolve `summonerId` to `puuid`."""
    url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/{summonerId}"
    response = requests.get(url, headers=getRiotHeaders())
    if response.status_code != 200:
        raise Exception(f"Failed to get summoner info: {response.status_code} - {response.text}")
    return response.json()["puuid"]


def sanitizeToken(value: str) -> str:
    """Convert a free-form token into a filename-safe token."""
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


def buildOutputPath(count: int, queue: str, region: str) -> Path:
    """Build per-run output file path under Data/players."""
    playersDir.mkdir(parents=True, exist_ok=True)
    queueToken = sanitizeToken(queue)
    regionToken = sanitizeToken(region)
    timestampToken = datetime.now().strftime("%Y%m%d-%H%M%S")
    fileName = f"top{count}-{queueToken}-{regionToken}-{timestampToken}.jsonl"
    return playersDir / fileName


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch top-ranked League players.")
    parser.add_argument("-n", "--count", type=int, default=300, help="Number of players to fetch")
    parser.add_argument("--region", default="na1", help="Regional routing (na1, euw1, kr, etc.)")
    parser.add_argument("--queue", default="RANKED_SOLO_5x5", help="Queue type")
    args = parser.parse_args()

    totalPlayers = 0
    writtenPlayers = 0
    seenPuuids: Set[str] = set()
    entries = getPlayers(n=args.count, region=args.region, queue=args.queue)
    outputPath = buildOutputPath(args.count, args.queue, args.region)

    with outputPath.open("w", encoding="utf-8") as outFile:
        for entry in entries:
            try:
                puuid = entry.get("puuid")
                if not puuid:
                    summonerId = entry.get("summonerId")
                    if not summonerId:
                        raise KeyError("Missing both 'puuid' and 'summonerId' in league entry")
                    puuid = getPuuidFromSummonerId(args.region, summonerId)
                    time.sleep(1)

                if puuid in seenPuuids:
                    print(f"Player with PUUID {puuid} already seen this run, skipping...")
                    totalPlayers += 1
                    continue

                riotId = getRiotId(puuid)
                time.sleep(1)

                playerRow = {
                    "username": riotId["gameName"],
                    "tag": riotId["tagLine"],
                    "puuid": puuid,
                    "sourceTier": entry.get("sourceTier", "unknown"),
                }
                outFile.write(json.dumps(playerRow, ensure_ascii=False) + "\n")
                outFile.flush()
                seenPuuids.add(puuid)
                writtenPlayers += 1

                print(
                    f"Fetched summoner: {riotId['gameName']}#{riotId['tagLine']} "
                    f"(tier={playerRow['sourceTier']}, puuid={puuid})"
                )
                totalPlayers += 1
            except Exception as error:
                summonerLabel = (
                    entry.get("summonerName")
                    or entry.get("summonerId")
                    or entry.get("puuid")
                    or "<unknown-entry>"
                )
                print(f"Error processing player {summonerLabel}: {error}")
                continue

    print(f"\nSuccessfully processed {totalPlayers} leaderboard entries")
    print(f"Wrote {writtenPlayers} players to {outputPath}")


if __name__ == "__main__":
    main()

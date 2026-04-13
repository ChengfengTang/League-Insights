"""
Process one match and write analysis logs to Data/log/<matchId>/.

How to run:
- python Data/processMatch.py --matchId NA1_5286644426
- python Data/processMatch.py  (process all matches under Data/matches)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


dataDir = Path(__file__).resolve().parent
matchesDir = dataDir / "matches"
timelinesDir = dataDir / "timelines"
logRootDir = dataDir / "log"


def msToMinSec(ms: int) -> str:
    """Convert milliseconds into MM:SS format."""
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"


def calculateDeathTimer(level: int, gameMinutes: float) -> int:
    """Calculate respawn timer from level and game time."""
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
    """Format champion label with team icon."""
    playerKey = str(playerId)
    champion = championMap.get(playerKey, {}).get("champion", f"Champion {playerKey}")
    team = championMap.get(playerKey, {}).get("team", "Unknown")
    if team == "Blue":
        return f"Blue {champion}"
    return f"Red {champion}"


def buildChampionMap(matchMeta: Dict) -> Dict[str, Dict[str, str]]:
    """Map participantId -> champion/team label."""
    championMap: Dict[str, Dict[str, str]] = {}
    for participant in matchMeta["info"]["participants"]:
        playerKey = str(participant["participantId"])
        championMap[playerKey] = {
            "champion": participant["championName"],
            "team": "Blue" if participant["teamId"] == 100 else "Red",
        }
    return championMap


def buildTimelineData(frames: List[Dict]) -> Dict[str, List[Dict]]:
    """Build per-participant timeline rows from frame data."""
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
    """Extract important events from frames."""
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
    """Create human-readable event log lines."""
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
    """Create human-readable movement log lines."""
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
    """Save text lines to file."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def processOneMatch(matchId: str) -> None:
    """Process one match id and write logs into Data/log/<matchId>/."""
    timelinePath = timelinesDir / f"{matchId}_timeline.json"
    matchPath = matchesDir / f"{matchId}.json"
    if not timelinePath.exists() or not matchPath.exists():
        return

    logDir = logRootDir / matchId
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a match into analysis logs under Data/log/<matchId>/")
    parser.add_argument("--matchId", required=False, help="Match id like NA1_5286644426")
    args = parser.parse_args()

    if args.matchId:
        processOneMatch(args.matchId)
        return

    for matchFile in sorted(matchesDir.glob("*.json")):
        matchId = matchFile.stem
        processOneMatch(matchId)


if __name__ == "__main__":
    main()

"""
Configuration for Live minimap monitor.
League of Legends Summoner's Rift uses in-game coordinates roughly 0–14820 (x) and 0–14881 (y).
The minimap is a scaled view of the full map (top-left minimap pixel = top-left of map).
"""

# Game map bounds (same scale as Riot API / timeline position data)
MAP_MIN_X = 0
MAP_MIN_Y = 0
MAP_MAX_X = 14820
MAP_MAX_Y = 14881

# Default minimap region (bottom-left of 1920x1080). Adjust for your resolution/UI scale.
# Format: (left, top, width, height). Top-left of screen is (0, 0).
DEFAULT_MINIMAP = {
    "left": 12,
    "top": 1080 - 12 - 256,  # 812: 12px from bottom, 256px height
    "width": 256,
    "height": 256,
}

# Enemy champion blobs on minimap are typically red. RGB range for detection.
ENEMY_RED_RGB_LOW = (140, 0, 0)
ENEMY_RED_RGB_HIGH = (255, 80, 80)

# Minimum blob area (pixels) to count as a champion; filters noise
MIN_BLOB_AREA = 15

# How often to capture and process (seconds)
CAPTURE_INTERVAL = 0.5

# Where to save captured positions (for training data)
CAPTURES_FILE = "live_captures.json"

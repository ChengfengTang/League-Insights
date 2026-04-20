# League Insights

A multi-module repository for League of Legends jungler analysis: machine learning model training, web-based match exploration, and live game monitoring with real-time predictions.

## Project structure

This repository includes a shared **`Data/`** pipeline (Riot data collection and offline processing) plus three application modules: **Predict**, **Replay**, and **Live**.

### **`Data/`** - Riot collection and offline match processing

Scripts live under `Data/` and are intended to be run from the **repository root**. Add Riot development keys to `Data/riotApiKey.txt` (one key per line; lines starting with `#` are treated as comments). **`fetchPlayer`** and **`fetchMatch`** require keys; **`processMatch`** is local/offline only. Large outputs (`Data/players/`, `Data/matches/`, `Data/timelines/`, `Data/log/`, and the key file) are listed in **`.gitignore`** so they are not committed to version control.

| Script | Role |
|--------|------|
| **`Data/fetchPlayer.py`** | Downloads ranked players from **Challenger through Diamond IV** for the given `--region` and `--queue`. Writes `Data/players/<region>-<queue>-<timestamp>/` with flat JSONL files (`challenger-grandmaster.jsonl`, `master.jsonl`, `diamondI.jsonl` through `diamondIV.jsonl`). Each line is JSON with **`username`** and **`tag`** (Riot ID); `fetchMatch` resolves **`puuid`** from the Account API when downloading matches. Uses **multiprocessing**: one worker process per API key, with a queue of tier jobs. |
| **`Data/fetchMatch.py`** | **`--playersFolder`** must point at a player run directory (for example `players/na1-ranked-solo-5x5-20260413-235836`). Loads every `*.jsonl` there, **merges unique players** (by Riot ID), and splits them across **one worker per API key**. Writes flat `Data/matches/<run>/` and `Data/timelines/<run>/`. Skips a match ID if any of `Data/matches/<run>/<matchId>.json`, `Data/timelines/<run>/<matchId>_timeline.json`, `Data/log/<run>/.done/<matchId>`, or `Data/log/<run>/<ChampionName>/<matchId>_p<participantId>_<side>.json` already exists. The Match v5 **routing cluster** (`americas`, `europe`, or `asia`) is inferred from the **first segment** of the folder name (for example `na1` in `na1-...`). Optional **`--count`** (default 100) limits how many match IDs are requested per player. |
| **`Data/processMatch.py`** | **Offline only.** Takes a single **positional argument**, the run name (for example `na1-ranked-solo-5x5-20260413-235836`). Reads flat `Data/matches/<runName>/*.json` with `Data/timelines/<runName>/`. Splits match files across up to **`CPU count`** worker processes (chunk jobs). Writes one pretty-printed JSON file per jungler under `Data/log/<runName>/<ChampionName>/<matchId>_p<participantId>_<blue|red>.json`, plus an empty `.done/<matchId>` marker to record successful processing. Each file contains a metadata header and a time-ordered `rows` array mixing frame rows with event rows such as `kill`, `assist`, `death`, `respawn`, `dragon`, `plate`, and purchase-driven `backing`. After a successful run, raw match and timeline files for that match are removed. |

**Bulk collection (rough planning numbers):** Expect on the order of **~20 completed matches per minute per development API key** once downloads are steady (each `matchId` needs match metadata plus timeline). The default **`--count 100`** caps how many match-history IDs are requested **per player**; cross-player **`matchId` deduplication** means the unique corpus is much smaller than “100 × number of players.” A full multi-ladder pass is sized around **~700,000 unique matches** at the intended scale.

**Typical flow:** run `fetchPlayer`, then `fetchMatch` using the same player run folder, then `processMatch` using the same directory name as under `matches/`. No extra champion-sorting step is needed because `processMatch` already writes champion-grouped output. The `Data/` scripts require the **`requests`** library (`pip install requests`) for anything that calls Riot.

**Script names:** The current pipeline uses **`Data/fetchPlayer.py`**, **`Data/fetchMatch.py`**, and **`Data/processMatch.py`**. Older notes or forks may mention flat `matchId.jsonl` logs, `sortLog.py`, or `Predict/loadData.py` as the main builder; those describe the older flow, not the current one.

### **`Predict/`** - Machine learning training

ML pipeline for jungler path prediction and analysis.

**Components:**

- **Training data**: The shared **`Data/`** pipeline is the primary dataset builder.
- **Research note**: `TD.txt` documents the current problem framing, dataset layout, and modeling decisions.
- **Champion categorization**: `champCategory.py` stores jungler-oriented playstyle labels and groupings.
- **Legacy placeholder**: `loadData.py` is no longer the main transformation step; grouped ML-ready logs are built directly by `Data/processMatch.py`.

**Data storage (when using the `Data/` pipeline):**

- Match and timeline JSON: `Data/matches/`, `Data/timelines/` (removed after successful `processMatch` for that match).
- Processed ML-ready match files: `Data/log/<run>/<ChampionName>/<matchId>_p<participantId>_<side>.json`, plus `Data/log/<run>/.done/<matchId>` markers.
- Player lists: `Data/players/`.
- Training scripts should be pointed at these paths as needed.

### **`Replay/`** - Web application (OP.GG-style)

Web application for match exploration and visualization.

**Components:**

- **Summoner lookup**: Search by Riot ID (name#tag).
- **Match browsing**: Recent matches with details.
- **Replay visualization**: Map view with champion movement over time.
- **Model inference API**: Planned read-only API for ML predictions.

**Architecture:**

- Flask backend.
- No database; each search uses the Riot API.
- Client-side visualization with p5.js.

### **`Live/`** - Live game monitor and prediction

Monitors the in-game minimap and provides jungler path predictions in real time.

**Components:**

- **Minimap monitor**: Screen capture and computer vision on the minimap.
- **Jungler detection**: Detects the jungler on the minimap and records coordinates.
- **Position collection**: Stores samples usable for training.
- **Model inference**: Loads models from `Predict/models/` when available.
- **AI assistant** (`assistant.py`): Terminal chat that receives game time, minimap-related context, and model output; you describe your champion and role and ask for jungle-tracking guidance.
- **Data pipeline**: Feeds captured data into the trained model stack.

**Architecture:**

- No Riot API for live capture; relies on screen content.
- Predict module supplies trained weights for inference.
- Predictions update as new positions are observed.

## Running the Data/ scripts

Use a terminal **in the repository root** (the directory that contains `Data/`, `README.md`, etc.). If `python` is not found on Windows, try `py` (for example `py Data/fetchPlayer.py`).

**Secrets:** `fetchPlayer` and `fetchMatch` read **`Data/riotApiKey.txt`**. Put one Riot development API key per line; lines starting with `#` are comments.

**Placeholders:** Replace example folder names (such as `na1-ranked-solo-5x5-20260413-235836`) with the real name printed by the previous step or shown in `Data/players/` / `Data/matches/`.

### fetchPlayer.py

Fetches ranked ladders and writes a new run under `Data/players/`.

| Argument | Required | Default | Meaning |
|----------|----------|---------|---------|
| `--region` | No | `na1` | Platform routing (examples: `euw1`, `kr`, `na1`). |
| `--queue` | No | `RANKED_SOLO_5x5` | Ranked queue id (example: `RANKED_FLEX_SR`). |

**Example:**

```bash
python Data/fetchPlayer.py --region na1 --queue RANKED_SOLO_5x5
```

**Output:** `Data/players/<region>-<queue>-<timestamp>/` with six **top-level** JSONL files (`challenger-grandmaster.jsonl`, `master.jsonl`, `diamondI.jsonl`, …). Copy the new folder name for the next step.

### fetchMatch.py

Downloads match metadata and timelines for every player listed in the JSONL files for one player run.

| Argument | Required | Default | Meaning |
|----------|----------|---------|---------|
| `--playersFolder` | **Yes** | (none) | Folder produced by `fetchPlayer`. Typical values: `players/<run-folder-name>` from the repo root, **or** just `<run-folder-name>` (the script looks under `Data/players/`), **or** an absolute path to the run folder. |
| `--count` | No | `100` | Maximum match IDs to request **per player**. |

The player run folder must contain at least one `*.jsonl` file **in that folder itself** (not only inside subfolders). All files are read; players are merged (duplicate Riot IDs dropped) then sharded across your API keys.

**Examples:**

```bash
python Data/fetchMatch.py --playersFolder players/na1-ranked-solo-5x5-20260413-235836
python Data/fetchMatch.py --playersFolder na1-ranked-solo-5x5-20260413-235836 --count 50
```

**Output:** `Data/matches/<run>/` and `Data/timelines/<run>/` (flat; no tier subfolders). The `<run>` folder name matches the player run basename.

### processMatch.py

Reads local match and timeline JSON only (no Riot traffic, no API key file needed for this step).

| Argument | Required | Meaning |
|----------|----------|---------|
| `runName` (positional) | **Yes** | The run folder name **only**, matching `Data/matches/<runName>/`. Example: `na1-ranked-solo-5x5-20260413-235836`. If you paste a path, only the **last path segment** is used. |

**Example:**

```bash
python Data/processMatch.py na1-ranked-solo-5x5-20260413-235836
```

**Output:** one pretty JSON file per jungler at `Data/log/<runName>/<ChampionName>/<matchId>_p<participantId>_<blue|red>.json`, plus an empty sentinel file at `Data/log/<runName>/.done/<matchId>`.

Each per-jungler file has:

- A top-level header with `matchId`, `participantId`, `teamId`, `isBlueSide`, `championId`, `championName`, `enemyChampionId`, and `enemyChampionName`.
- A chronological `rows` array covering the laning phase. `rows` mixes regular `frame` rows with event rows such as `kill`, `assist`, `death`, `respawn`, `dragon`, `plate`, and `backing`.
- A unique sequential `frameIndex` across the merged sequence.
- `targetX`, `targetY`, `targetDx`, `targetDy`, and `targetHorizonMs` on every non-terminal row, pointing to the **next chronological row** in the merged sequence rather than only the next fixed 60-second frame.

`backing` rows are sourced from timeline `ITEM_PURCHASED` events and nearby purchases are collapsed into a single shop visit. `respawn` rows are inferred from death time and level so the sequence can model `death -> respawn -> next frame`. The `.done` files are intentionally empty: they are only success markers used for idempotency.

### End-to-end command list

Use one consistent run name (here `na1-ranked-solo-5x5-20260413-235836` is only an illustration; your `fetchPlayer` run will have its own timestamp):

```bash
pip install requests
python Data/fetchPlayer.py --region na1 --queue RANKED_SOLO_5x5
python Data/fetchMatch.py --playersFolder players/na1-ranked-solo-5x5-20260413-235836
python Data/processMatch.py na1-ranked-solo-5x5-20260413-235836
```

## Features

### Summoner lookup (Replay)

- Search by Riot ID (name#tag).
- API key handling for Replay.
- Recent match lists.

### Match history (Replay)

- Match details, mode, duration.
- Copy match ID.
- Team composition display.

### Replay system (Replay)

- Interactive map and movement playback.
- Events: kills, assists, deaths, respawns, epic monsters, level-ups.
- Death timer logic using level, game time, base respawn window, and time impact factor.

### Data pipeline (`Data/`)

- Ranked ladder coverage from Challenger through Diamond IV, with optional multi-key parallelism in `fetchPlayer`.
- Bulk match and timeline files under flat `matches/<run>/` and `timelines/<run>/` from `fetchMatch`, with routing derived from the run folder prefix.
- `processMatch` builds grouped per-jungler logs directly under `log/<run>/<ChampionName>/`, adds `.done` markers for processed matches, and deletes raw match/timeline pairs after successful processing.
- Event-aware rows include `kill`, `assist`, `death`, `respawn`, `dragon`, `plate`, and purchase-driven `backing`, not just coarse per-frame aggregates.

### Data processing (Predict)

- Per-timestamp player stats: position, level, CS, gold.
- Events: champion kills, assists, epic monsters, level-ups, deaths and respawns.
- Champion playstyle categories.
- Enemy-jungler pathing focus with laning-phase-first workflows and optional clear-path priors.
- Model training and optional category-specific models.

### Live game monitoring (Live)

- Minimap-driven capture and jungler-focused inference.
- Optional **in-game AI assistant** (OpenAI-compatible or local Ollama): describe your pick and ask for jungle tracking help.
- Training value from captured minimap positions.
- Integration with Predict model artifacts.

## Technical stack

### Data

- **Language**: Python 3.
- **APIs**: Riot League, Summoner, and Account (in `fetchPlayer`); Match v5 (in `fetchMatch`).
- **Storage**: `Data/players/`, `Data/matches/`, `Data/timelines/`, `Data/log/` (see `.gitignore`).
- **Libraries**: `requests`; `processMatch` is filesystem-only.

### Replay

- **Backend**: Python (Flask).
- **Frontend**: HTML, JavaScript, p5.js.
- **APIs**: Riot Match v5.
- **Storage**: No database; live API calls per request.

### Predict

- **Language**: Python.
- **APIs**: Riot Match v5 (and related as used in scripts).
- **Storage**: Local disk; bulk artifacts often under `Data/matches/`, `Data/timelines/`, `Data/log/` when using the shared `Data/` scripts.
- **ML**: scikit-learn (decision tree, gradient boosting).
- **Libraries**: pandas, numpy, scikit-learn.

### Live

- **Language**: Python.
- **Data sources**: Screen capture and CV only (no Riot API for live frames).
- **Libraries**: mss, OpenCV (minimap analysis), numpy, OpenAI client (optional, for assistant).
- **Entry points**: `python -m Live.live_monitor`, `python -m Live.assistant` (module paths; files live under `Live/`).
- **Models**: Loads from `Predict/models/` when configured.
- **Captures**: `Live/live_captures.json`.

## Setup

Install only what you need for the module you are working on. The checked-in `requirements.txt` file today is `Replay/requirements.txt`; the **`Data/`** scripts additionally need **`requests`** (install globally in your environment or in the same virtual environment you use for training). From the repository root:

```bash
pip install -r <Module>/requirements.txt
```

Use `Replay`, `Predict`, or `Live` in place of `<Module>`.

### Prerequisites

1. Riot API key from the [Riot Developer Portal](https://developer.riotgames.com/) (for any module or script that calls Riot).
2. Python 3.7 or newer.
3. (Optional) MySQL, only if you enable Predict database features that require it.

### Replay setup

1. Install dependencies:

   ```bash
   pip install -r Replay/requirements.txt
   ```

2. Start the app from the repository root or after `cd Replay`:

   ```bash
   cd Replay && python application.py
   ```

3. Open `http://127.0.0.1:5000` in a browser.

### Data pipeline (optional, repository root)

1. Install `requests` and create `Data/riotApiKey.txt` with your key(s) (see **Running the Data/ scripts** above for flags, path rules, and examples).

2. Run in order: `Data/fetchPlayer.py`, then `Data/fetchMatch.py --playersFolder ...`, then `Data/processMatch.py <runName>`.

### Predict setup

1. Create or activate your preferred Python environment for ML work.

2. (Recommended) Refresh training data using the **Data pipeline** steps above so `Data/processMatch.py` generates the current grouped log format.

3. Point training code at `Data/log/<run>/<ChampionName>/` or walk the full `Data/log/<run>/` tree if you want all champions.

4. Use `Predict/TD.txt` as the reference for the expected row schema, event types, and modeling assumptions.

### Live setup

1. Install dependencies:

   ```bash
   pip install -r Live/requirements.txt
   ```

   Includes `mss` and `opencv-python-headless` for capture and blob detection.

2. (Optional) Train or copy models into `Predict/models/` for prediction mode.

3. (Optional) Calibrate the minimap region from the repository root:

   ```bash
   python -m Live.calibrate
   ```

   This writes `Live/minimap_calibrate.png`. Adjust `Live/config.py` (`DEFAULT_MINIMAP`) or pass `--left`, `--top`, `--width`, `--height` when starting the monitor.

4. Run the monitor:

   ```bash
   python -m Live.live_monitor
   ```

   Use `--no-predict` to record positions only; `--predictor lstm` or `--predictor tree` to select a backend. Captures are written to `Live/live_captures.json`.

**AI assistant (terminal chat with live context)**

With an OpenAI-compatible API (hosted OpenAI or a local stack such as Ollama):

```bash
export OPENAI_API_KEY="your-key"   # for Ollama you might set OPENAI_BASE_URL instead, e.g. http://localhost:11434/v1
python -m Live.assistant
```

Type questions in the terminal (for example which lane or champion you are playing). Use `quit` or `exit` to stop, `clear` to reset context. Optional: `OPENAI_MODEL` (default is often `gpt-4o-mini` depending on configuration).

## Notes

- **Replay** does not use a database; each lookup hits the Riot API.
- **Data/** groups ranked snapshots, bulk match and timeline download, and offline log generation. Generated folders are gitignored; never commit API keys.
- **Predict** should consume the grouped per-jungler files produced by **`Data/processMatch.py`**. `Predict/loadData.py` is legacy and no longer defines the main on-disk training format.
- **Live** benefits from trained models in `Predict/models/` but can run in capture-only modes.
- **Live** does not call the Riot API for live frames; it uses the screen.
- **Predict**, **Replay**, and **Live** can be used on their own in many cases; **Live** depends on **Predict** for bundled model files when prediction is enabled.
- A future Replay API may read Predict artifacts directly (not yet implemented).
- **Live** targets active games; **Replay** targets finished matches in the browser.

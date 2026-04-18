# League Insights

A multi-module repository for League of Legends jungler analysis: machine learning model training, web-based match exploration, and live game monitoring with real-time predictions.

## Project structure

This repository includes a shared **`Data/`** pipeline (Riot data collection and offline processing) plus three application modules: **Predict**, **Replay**, and **Live**.

### **`Data/`** - Riot collection and offline match processing

Scripts live under `Data/` and are intended to be run from the **repository root**. Add Riot development keys to `Data/riotApiKey.txt` (one key per line; lines starting with `#` are treated as comments). **`fetchPlayer`** and **`fetchMatch`** require keys; **`processMatch`** does not contact Riot. Large outputs (`Data/players/`, `matches/`, `timelines/`, `log/`, and the key file) are listed in **`.gitignore`** so they are not committed to version control.

| Script | Role |
|--------|------|
| **`Data/fetchPlayer.py`** | Downloads ranked players from **Challenger through Diamond IV** for the given `--region` and `--queue`. Writes `Data/players/<region>-<queue>-<timestamp>/` with flat JSONL files (`challenger-grandmaster.jsonl`, `master.jsonl`, `diamondI.jsonl` through `diamondIV.jsonl`). Each line contains `username`, `tag`, `puuid`, and `sourceTier`. Uses **multiprocessing**: one worker process per API key, with a queue of tier jobs. |
| **`Data/fetchMatch.py`** | **`--playersFolder`** must point at a player run directory (for example `players/na1-ranked-solo-5x5-20260413-235836`). Loads every `*.jsonl` there, **merges unique players** (by Riot ID), and splits them across **one worker per API key**. Writes flat `Data/matches/<run>/` and `Data/timelines/<run>/`. The Match v5 **routing cluster** (`americas`, `europe`, or `asia`) is inferred from the **first segment** of the folder name (for example `na1` in `na1-...`). Optional **`--count`** (default 100) limits how many match IDs are requested per player. |
| **`Data/processMatch.py`** | **Offline only.** Takes a single **positional argument**, the run name (for example `na1-ranked-solo-5x5-20260413-235836`). Reads flat `Data/matches/<runName>/*.json` with `Data/timelines/<runName>/`. Splits match files across up to **`CPU count`** worker processes. Writes `Data/log/<runName>/<matchId>.jsonl`. |

**Typical flow:** run `fetchPlayer`, then `fetchMatch` using the same player run folder, then `processMatch` using the same directory name as under `matches/`. The `Data/` scripts require the **`requests`** library (`pip install requests`).

**Script names:** The current pipeline uses only **`Data/fetchPlayer.py`**, **`Data/fetchMatch.py`**, and **`Data/processMatch.py`**. Older notes or forks may mention names such as **`fetchTopRankedPlayers.py`**; that is not part of this layout anymore.

### **`Predict/`** - Machine learning training

ML pipeline for jungler path prediction and analysis.

**Components:**

- **Training data**: Prefer the **`Data/`** pipeline above for bulk `players/`, `matches/`, `timelines/`, and `log/`; Predict's **`fetchdata.py`** remains available for ad-hoc or legacy workflows.
- **Timeline parsing**: `AllInfo.py` - match timeline analysis and data processing.
- **Data fetching**: `fetchdata.py` - Riot API utilities.
- **Champion categorization**: `TDchampCategory.py` - playstyle labels (for example aggressive, full_clear).
- **Model training**: `TDpredict.py` - models for enemy jungler location prediction.
- **YOLO champion icons** (`Predict/yolo/`): Data Dragon icons, synthetic YOLO data, Ultralytics YOLOv8 training; `Live/yolo_detect.py` runs on minimap crops (see `Predict/yolo/README.md`).
- **Model artifacts**: Intended location `models/` inside Predict.

**Data storage (when using the `Data/` pipeline):**

- Match and timeline JSON: `Data/matches/`, `Data/timelines/`.
- Processed ML-ready match files: `Data/log/`.
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

**Output:** `Data/log/<runName>/<matchId>.jsonl`. Each file contains one parsed match object with three sections:

- `matchContext`: static match-level metadata that does not change over time, including `matchId`, patch, queue/map, both teams, all participants, and the identified blue/red junglers.
- `junglerTrainingRows`: one row per jungler per frame with only online-safe movement features, including position, short motion history, level/xp/gold/CS/HP state, recent nearby pressure from events, and next-frame targets used for supervised training.
- `events`: only coordinate-bearing events involving a jungler, mainly spatial `CHAMPION_KILL` and `ELITE_MONSTER_KILL` rows. These help explain rotations without including non-spatial noise such as `LEVEL_UP`.

This extraction is designed specifically for jungler movement prediction. Match metadata provides static role/champion/context, timeline rows provide the per-timestamp training features at time `t`, and the filtered spatial events provide nearby fight/objective pressure that often drives pathing decisions.

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
- `processMatch` builds `log/<run>/<matchId>.jsonl` from local files only, parallelized by CPU count.

### Data processing (Predict)

- Per-timestamp player stats: position, level, CS, gold.
- Events: champion kills, assists, epic monsters, level-ups, deaths and respawns.
- Champion playstyle categories.
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

Each module ships a **`requirements.txt`**. Install only what you need: Replay (web), Predict (ML), Live (capture and optional models). The **`Data/`** scripts additionally need **`requests`** (install globally in your environment or in the same virtual environment you use for Predict). From the repository root:

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

2. Run the three commands in order: `Data/fetchPlayer.py`, then `Data/fetchMatch.py --playersFolder ...`, then `Data/processMatch.py <runName>`.

### Predict setup

1. Install dependencies:

   ```bash
   pip install -r Predict/requirements.txt
   ```

2. If Predict scripts expect it, copy `.env.example` to `.env` (or follow project conventions) and set `RIOT_API_KEY=...`.

3. (Recommended) Refresh training data using the **Data pipeline** steps above, or continue using Predict fetch utilities if your workflow depends on them.

4. Work inside Predict:

   ```bash
   cd Predict
   ```

5. Point `fetchdata.py` or other tools at `Data/matches/` and `Data/timelines/` as appropriate.

6. Run timeline analysis:

   ```bash
   python AllInfo.py
   ```

7. Train models:

   ```bash
   python TDpredict.py
   ```

   Outputs typically go under `models/`. Live can consume these artifacts for real-time prediction.

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
- **Predict** stores training data on disk. Prefer **`Data/fetchPlayer.py`**, **`Data/fetchMatch.py`**, and **`Data/processMatch.py`** for repeatable runs, or **`fetchdata.py`** and related tools for legacy flows.
- **Live** benefits from trained models in `Predict/models/` but can run in capture-only modes.
- **Live** does not call the Riot API for live frames; it uses the screen.
- **Predict**, **Replay**, and **Live** can be used on their own in many cases; **Live** depends on **Predict** for bundled model files when prediction is enabled.
- A future Replay API may read Predict artifacts directly (not yet implemented).
- **Live** targets active games; **Replay** targets finished matches in the browser.

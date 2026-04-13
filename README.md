# 🧠 League Insights

A multi-module repository for League of Legends jungler analysis: machine learning model training, web-based match exploration, and live game monitoring with real-time predictions.

## 📁 Project Structure

This repository is organized into three independent modules:

### **`Predict/`** - Machine Learning Training Project
ML pipeline for jungler path prediction and analysis.

**Components:**
- **Shared Data Ingestion**: `Data/fetchTopRankedPlayers.py` - Fetches top ranked players (Challenger, Grandmaster, Master) for training/replay data collection
- **Timeline Parsing**: `AllInfo.py` - Match timeline analysis and data processing
- **Data Fetching**: `fetchdata.py` - Riot API data fetching utilities
- **Champion Categorization**: `TDchampCategory.py` - Categorizes champions by jungler playstyle (aggressive, full_clear, etc.)
- **Model Training**: `TDpredict.py` - ML model training and prediction for enemy jungler location prediction
- **YOLO champion icons** (`Predict/yolo/`): Download Data Dragon icons, synthetic YOLO dataset, train Ultralytics YOLOv8; `Live/yolo_detect.py` runs inference on minimap crops (see `Predict/yolo/README.md`)
- **Model Artifacts**: (To be stored in `models/` directory)

**Data Storage:**
- Fetches and stores match data locally in `timelines/` and `matches/` directories
- Used for training and testing ML models

### **`Replay/`** - Web Application (OP.GG-style)
OP.GG-style web application for match exploration and visualization.

**Components:**
- **Summoner Lookup**: Search for players by Riot ID (name#tag)
- **Match Browsing**: View recent matches with detailed information
- **Replay Visualization**: Interactive map visualization with real-time champion movement tracking
- **Model Inference API**: Read-only API for ML model predictions (to be implemented)

**Architecture:**
- Flask-based web application
- No database - every search fetches fresh data from Riot API
- Client-side visualization using p5.js

### **`Live/`** - Live Game Monitor & Prediction System
Real-time game monitoring system that visually watches the minimap and provides instant jungler path predictions.

**Components:**
- **Minimap Monitor**: Visually watches the game minimap using screen capture/computer vision
- **Jungler Detection**: Detects when the jungler appears on the minimap and captures coordinates
- **Position Data Collection**: Records captured position information for training data
- **Model Inference Engine**: Loads trained models from `Predict/models/` for live predictions
- **AI Assistant** (`assistant.py`): Text chat with an AI that sees game time, minimap positions, and model prediction—tell it what you're playing and ask for jungle-tracking help
- **Data Pipeline**: Processes captured minimap data and feeds it into the trained ML model

**Architecture:**
- Visual minimap monitoring (screen capture/computer vision) - no Riot API for live data
- Real-time coordinate capture when jungler appears on minimap
- Captured position data can be used for training (counts as position info)
- Model inference using trained artifacts from Predict module
- Continuous prediction updates as new position data is captured
- Integration with Predict module's trained models

## 🌟 Features

### 🔍 Summoner Lookup
- Search for players by their Riot ID (name#tag)
- Secure API key handling
- Recent matches retrieval

### 📊 Match History
- View recent matches with detailed information
- Game mode display
- Match duration formatting
- Copy match ID functionality
- Team composition visualization

### 🎮 Replay System
- Interactive map visualization
- Real-time champion movement tracking
- Event-based interpolation for:
  - Kills and assists
  - Deaths and respawns
  - Monster kills
  - Level ups
- Accurate death timer calculation based on:
  - Champion level
  - Game time
  - Base Respawn Window (BRW)
  - Time Impact Factor (TIF)

### 📈 Data Processing (Predict)
- Per-minute snapshots of player stats:
  - Position (x, y)
  - Level
  - CS (minions + jungle)
  - Gold
- Event tracking:
  - Champion kills and assists
  - Elite monster kills (Dragon, Herald, Baron)
  - Level ups
  - Deaths and respawns
- Champion categorization by playstyle (aggressive, full_clear, etc.)
- ML model training for enemy jungler location prediction
- Category-specific model training for improved accuracy

### 🎯 Live Game Monitoring (Live)
- Visual minimap monitoring using screen capture/computer vision
- Automatic jungler detection when they appear on the minimap
- Real-time coordinate capture from minimap positions
- **In-game AI assistant**: Text chat—tell the AI what you're playing; it sees time, minimap data, and the trained prediction and helps with jungle tracking (OpenAI or Ollama)
- Captured position data used for training (counts as position info)
- Continuous model inference using trained ML models
- Real-time jungler path predictions as the game progresses
- Integration with Predict module's trained models

## 🛠️ Technical Stack

### Replay
- **Backend**: Python (Flask)
- **Frontend**: HTML, JavaScript, p5.js
- **APIs**: Riot Games API (match-v5)
- **Storage**: No database - fresh API calls on each request

### Predict
- **Language**: Python
- **APIs**: Riot Games API (match-v5)
- **Storage**: Local file system (`timelines/`, `matches/` directories)
- **ML Framework**: scikit-learn (Decision Tree, Gradient Boosting)
- **Libraries**: pandas, numpy, scikit-learn

### Live
- **Language**: Python
- **Data Sources**: Visual minimap monitoring (screen capture/computer vision) - no Riot API
- **Libraries**: mss (screen capture), OpenCV (minimap analysis), numpy, openai (for AI assistant)
- **Scripts**: `live_monitor.py` (capture + prediction), `assistant.py` (text chat with AI)
- **Model Integration**: Loads trained models from `Predict/models/` for live predictions
- **Data Collection**: Records saved to `Live/live_captures.json`

## 🔧 Setup

Each module has its own **`requirements.txt`** so you only install what you need (Replay = web only; Predict = ML; Live = minimap capture + optional Predict models). Run `pip install -r <Module>/requirements.txt` from the project root or from inside that module’s directory.

### Prerequisites
1. Get a Riot API key from [Riot Developer Portal](https://developer.riotgames.com/)
2. Python 3.7+
3. (Optional) MySQL - only needed if you want to use database features in Predict

### Replay Setup
1. Install dependencies (from project root or Replay directory):
   ```bash
   pip install -r Replay/requirements.txt
   ```
2. Navigate to Replay and run the application:
   ```bash
   cd Replay && python application.py
   ```
3. Open your browser to `http://127.0.0.1:5000`

### Predict Setup
1. Install dependencies (from project root or Predict directory):
   ```bash
   pip install -r Predict/requirements.txt
   ```
2. Fill in your Riot API key in `.env.example` (`RIOT_API_KEY=...`).
3. Fetch training data (from project root):
   ```bash
   python Data/fetchTopRankedPlayers.py  # Fetches top players
   ```
4. Navigate to the ML project directory:
   ```bash
   cd Predict
   ```
5. Fetch match/timeline data:
   ```bash
   python fetchdata.py    # Fetches match data for training
   ```
6. Run timeline analysis:
   ```bash
   python AllInfo.py
   ```
7. Train prediction models:
   ```bash
   python TDpredict.py  # Trains models for enemy jungler location prediction
   ```
   - Models will be saved to `models/` directory
   - Supports category-specific models (aggressive, full_clear, etc.)
   - Model artifacts will be used by Live module for real-time predictions

### Live Setup
1. From the project root, install Live dependencies:
   ```bash
   pip install -r Live/requirements.txt
   ```
   (Uses `mss` for screen capture, `opencv-python-headless` for blob detection.)
2. Ensure Predict module has trained models in `Predict/models/` directory (optional, for predictions).
3. (Optional) Calibrate minimap region: run `python -m Live.calibrate` from project root; it saves `Live/minimap_calibrate.png`. Adjust `Live/config.py` (DEFAULT_MINIMAP) or use `--left`, `--top`, `--width`, `--height` when running the monitor.
4. Run the live monitor (from project root):
   ```bash
   python -m Live.live_monitor
   ```
   Use `--no-predict` to only record positions; `--predictor lstm` or `--predictor tree` to choose model. Captures are saved to `Live/live_captures.json`.

**AI Assistant (text chat with live context)**  
With an OpenAI-compatible API (OpenAI or local e.g. Ollama):
   ```bash
   export OPENAI_API_KEY="your-key"   # or for Ollama: export OPENAI_BASE_URL="http://localhost:11434/v1"
   python -m Live.assistant
   ```
   Type in the terminal: e.g. "I'm playing mid Ahri" or "Where is their jungler?" Use `quit` or `exit` to stop; `clear` to clear context. Optional: `OPENAI_MODEL` (default `gpt-4o-mini`).

## 📝 Notes

- **Replay** is stateless - no database required. Every search fetches fresh data from Riot API.
- **Predict** stores data locally for training purposes. Data is fetched using `Data/fetchTopRankedPlayers.py` and `fetchdata.py`.
- **Live** requires trained models from Predict module. Ensure models are trained and saved in `Predict/models/` before running Live.
- **Live** does not use Riot API - it visually monitors the minimap using screen capture/computer vision.
- When the jungler appears on the minimap, Live captures their coordinates, which counts as position info for training.
- All three modules can run independently, but Live depends on Predict for model artifacts.
- Model inference API in Replay will read model artifacts from Predict (to be implemented).
- Live module provides real-time predictions during active games, while Replay visualizes completed matches.


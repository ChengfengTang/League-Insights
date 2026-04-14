"""
League of Legends Replay Application

OP.GG-style web application for League of Legends match exploration.
This application provides functionality to:
1. Look up League of Legends summoners by their name and tag
2. Retrieve and display recent match history
3. Fetch detailed match data and timelines (fresh fetch, no database)
4. Visualize match replays and statistics
5. Model inference API (read-only) for ML predictions

The application uses:
- Flask for the web framework
- Riot Games API for League of Legends data
- Environment variables for configuration
- No database - all data fetched fresh on each request
"""

from flask import Flask, request, jsonify, send_file, render_template
import requests
import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file for secure configuration
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

MATCH_REGION = "americas" # May need to add options for EUROPE and ASIA but doesn't seem to matter right now
DEFAULT_MATCH_COUNT = 5
DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent.parent / "Data" / "riotApiKey.txt"

def resolve_api_key(request_api_key: Optional[str]) -> Optional[str]:
    """
    Resolve Riot API key by priority:
    1) query param api_key
    2) RIOT_API_KEY env var
    3) last non-empty line in Data/riotApiKey.txt
    """
    if request_api_key and request_api_key.strip():
        return request_api_key.strip()

    env_key = os.getenv("RIOT_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        if DEFAULT_API_KEY_PATH.exists():
            lines = [line.strip() for line in DEFAULT_API_KEY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                return lines[-1]
    except OSError:
        return None

    return None

@app.route("/api/lookup")
def lookup():
    """
    API endpoint to look up a League of Legends summoner by name and tag.
    
    Query Parameters:
        name (str): Summoner's name
        tag (str): Summoner's tag (e.g., NA1)
        api_key (str): Riot Games API key
    
    Returns:
        JSON response containing the summoner's PUUID and other account information
    """
    name = request.args.get("name")
    tag = request.args.get("tag")
    api_key = resolve_api_key(request.args.get("api_key"))
    
    # Validate required parameters
    if not name or not tag:
        return jsonify({"error": "Missing name or tag"}), 400
    if not api_key:
        return jsonify({"error": "Missing API key (query, RIOT_API_KEY, or Data/riotApiKey.txt)"}), 400

    # Set up API request headers
    headers = {'X-Riot-Token': api_key}
    url = f"https://{MATCH_REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
    r = requests.get(url, headers=headers)
    # If the request fails (status code not 200), returns an error.
    # If successful, returns the API's JSON response (e.g., {"puuid": "abc123", ...}).
    if r.status_code == 429:
        return jsonify({"error": "Rate limit exceeded, try again later"}), 429
    elif r.status_code != 200:
        return jsonify({"error": f"Failed to fetch PUUID: {r.status_code}"}), r.status_code
    return jsonify(r.json())

@app.route("/api/matches/<puuid>")
def get_matches(puuid):
    """
    API endpoint to retrieve recent matches for a summoner.
    
    Parameters:
        puuid (str): Player's unique identifier
    
    Query Parameters:
        api_key (str): Riot Games API key
        start (int): Starting index for pagination (default: 0)
        count (int): Number of matches to retrieve (default: 5)
    
    Returns:
        JSON array of match IDs
    """
    api_key = resolve_api_key(request.args.get("api_key"))
    if not api_key:
        return jsonify({"error": "Missing API key (query, RIOT_API_KEY, or Data/riotApiKey.txt)"}), 400

    # Parse and validate pagination parameters
    start = request.args.get("start", "0")
    count = request.args.get("count", str(DEFAULT_MATCH_COUNT))
    
    try:
        start = int(start)
        count = int(count)
    except ValueError:
        return jsonify({"error": "Invalid start or count parameters"}), 400

    # Fetch matches from Riot API
    headers = {'X-Riot-Token': api_key}
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
    r = requests.get(url, headers=headers)
    
    # Handle API response
    if r.status_code == 429:
        return jsonify({"error": "Rate limit exceeded, try again later"}), 429
    elif r.status_code != 200:
        return jsonify({"error": f"Failed to fetch matches: {r.status_code}"}), r.status_code
    
    return jsonify(r.json())

@app.route("/api/matches/<puuid>/more")
def get_more_matches(puuid):
    """
    API endpoint to load additional matches for pagination.
    Similar to get_matches but specifically for loading more matches.
    
    Parameters and functionality are identical to get_matches.
    """
    api_key = resolve_api_key(request.args.get("api_key"))
    if not api_key:
        return jsonify({"error": "Missing API key (query, RIOT_API_KEY, or Data/riotApiKey.txt)"}), 400

    start = request.args.get("start", "0")
    count = request.args.get("count", str(DEFAULT_MATCH_COUNT))
    
    try:
        start = int(start)
        count = int(count)
    except ValueError:
        return jsonify({"error": "Invalid start or count parameters"}), 400

    headers = {'X-Riot-Token': api_key}
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}"
    r = requests.get(url, headers=headers)
    if r.status_code == 429:
        return jsonify({"error": "Rate limit exceeded, try again later"}), 429
    elif r.status_code != 200:
        return jsonify({"error": f"Failed to fetch matches: {r.status_code}"}), r.status_code
    
    return jsonify(r.json())

@app.route("/api/match/<match_id>")
def get_match_data(match_id):
    """
    API endpoint to retrieve detailed match data and timeline.
    
    Parameters:
        match_id (str): Unique identifier for the match
    
    Query Parameters:
        api_key (str): Riot Games API key
    
    Returns:
        JSON object containing match metadata and timeline data
    """
    api_key = resolve_api_key(request.args.get("api_key"))
    if not api_key:
        return jsonify({"error": "Missing API key (query, RIOT_API_KEY, or Data/riotApiKey.txt)"}), 400

    # Always fetch fresh data from Riot API (no database caching)
    headers = {'X-Riot-Token': api_key}
    
    # Get match data
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return jsonify({"error": f"Failed to fetch match metadata: {r.status_code}"}), r.status_code
    match_data = r.json()
    
    # Get timeline data
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return jsonify({"error": f"Failed to fetch timeline: {r.status_code}"}), r.status_code
    timeline_data = r.json()
    
    return jsonify({"metadata": match_data, "timeline": timeline_data})

# Route handlers for serving HTML pages
@app.route('/')
def home():
    """Serve the main summoner lookup page"""
    return render_template('SummonerLookup.html')

# Serve matches page
@app.route('/matches.html')
def matches_page():
    """Serve the matches history page"""
    return render_template('matches.html')

# Serve replay page
@app.route('/replay.html')
def replay_page():
    """Serve the match replay visualization page"""
    return render_template('replay.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
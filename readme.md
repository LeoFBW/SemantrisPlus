# LLM-Powered Semantris Tower

Lightweight, Flask-based reimagining of Google’s Semantris. You type a clue, an LLM re-ranks the tower of words by semantic relevance, and the lowest-ranked “target” word causes a mini Tetris-like collapse. The front end is a single Tailwind-powered `arcade.html`; the back end offers two variants:

- `app.py`: Gemini-only implementation.
- `app2(OpenAI).py`: Pluggable LLM version with Gemini, OpenAI, or a custom OpenAI-compatible endpoint.

## Project tour
- `templates/arcade.html`: Game UI, FLIP-style animations, keyboard hook, and fetch calls to `/rank`.
- `app.py`: Gemini-only gameplay loop and vocabulary loader (`assets/The_digital_space_1.txt` by default).
- `app2(OpenAI).py`: Same gameplay, but lets you switch between Gemini, OpenAI, or a custom endpoint via env vars. Uses `assets/worldwide_destinations_1.txt` by default.
- `assets/*.txt`: Word lists; swap or add your own to change themes.
- `requirements.txt`: Flask, google-generativeai, openai, python-dotenv, etc.

## Setup
1) Install dependencies
```
pip install -r requirements.txt
```

2) Create a `.env` in the project root
```
FLASK_SECRET_KEY="set-a-random-secret"

# Pick one LLM path:
# Gemini (used by app.py, or app2 when LLM_PROVIDER=gemini)
GEMINI_API_KEY="your-gemini-api-key"

# OpenAI (used only by app2 when LLM_PROVIDER=openai)
OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL_NAME="gpt-4o-mini"   # optional override

# Custom OpenAI-compatible endpoint (used only by app2 when LLM_PROVIDER=custom)
CUSTOM_ENDPOINT_URL="https://your-endpoint/v1"
CUSTOM_API_KEY="your-custom-api-key"
CUSTOM_MODEL_NAME="your-model"    # required for custom mode

# app2 only: choose which provider to use
LLM_PROVIDER="gemini|openai|custom"
```

## Run the game
- Gemini-only version:
```
python app.py
```

- Switchable provider version (defaults to `LLM_PROVIDER=custom` unless you override):
```
python "app2(OpenAI).py"
```

The server runs on `http://127.0.0.1:5001/`. Open it in your browser.

## How to play
- A tower of words is shown; one is the current **target**.
- Type a clue and press Enter. The LLM reorders the tower by relevance to your clue.
- If the target lands in the bottom four positions, those words disappear; your score increases and new words drop in.
- The board size grows with score; you win when the vocabulary runs out and no new words can be added.

## Customizing vocabulary
- Each app reads from its configured text file under `assets/`.
- One word per line. Empty lines are ignored.
- If the file is missing or empty, a built-in fallback list is used.

## Notes and troubleshooting
- If the LLM API call fails, `/rank` returns the current board unchanged and sets `api_error=true` for the front end to display.
- Ensure your `.env` matches the file you run (`app.py` needs Gemini; `app2(OpenAI).py` needs the provider-specific keys).
- `FLASK_SECRET_KEY` is required for session state; set it to any random string in development.

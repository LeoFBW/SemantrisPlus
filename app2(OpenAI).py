import os
import time
import random
import google.generativeai as genai
import openai  # <<< NEW >>>
import requests  # <<< NEW >>>
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

# --- 0. Environment & Vocabulary ---

VOCABULARY = []
VOCAB_FILE = os.path.join('assets', 'The_digital_space_1.txt')

# Load environment variables from .env file
load_dotenv()

# --- 1. Configuration ---

app = Flask(__name__)
# SECRET_KEY is required for Flask sessions.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# <<< NEW: Multi-LLM Configuration >>>
# This section loads all necessary keys and settings from your .env file
# Your .env file should now look something like this:
#
# # --- General ---
# FLASK_SECRET_KEY="your_flask_secret_key"
#
# # --- Provider Choice ---
# # (gemini, openai, or custom)
# LLM_PROVIDER="gemini"
#
# # --- Gemini ---
# GEMINI_API_KEY="your_gemini_api_key"
#
# # --- OpenAI ---
# OPENAI_API_KEY="your_openai_api_key"
# OPENAI_MODEL_NAME="gpt-4o-mini"
#
# # --- Custom Endpoint ---
# # (An example for a self-hosted model)
# CUSTOM_ENDPOINT_URL="http://localhost:11434/api/generate"
# CUSTOM_API_KEY="your_custom_api_key" # Optional, for "Authorization: Bearer"
# CUSTOM_MODEL_NAME="llama3"

# --- Load LLM Provider Settings ---
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'custom').lower()

# --- Gemini Settings ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- OpenAI Settings ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")

# --- Custom Endpoint Settings ---
CUSTOM_ENDPOINT_URL = os.environ.get("CUSTOM_ENDPOINT_URL")
CUSTOM_API_KEY = os.environ.get("CUSTOM_API_KEY")  # For Bearer token auth
CUSTOM_MODEL_NAME = os.environ.get("CUSTOM_MODEL_NAME") # Optional, for custom payload


# --- 2. Game Content ---
def calculate_board_size(score):
    """
    Minimum 5, maximum 20.
    Board size = min(20, max(5, score//2))
    """
    size = max(5, score // 2)
    return min(size, 22)

def load_vocabulary():
    """Loads the vocabulary from the assets file."""
    global VOCABULARY
    try:
        with open(VOCAB_FILE, 'r') as f:
            VOCABULARY = [line.strip() for line in f if line.strip()]
        if not VOCABULARY:
            raise FileNotFoundError("Vocabulary file is empty.")
        print(f"Successfully loaded {len(VOCABULARY)} words.")
    except FileNotFoundError:
        print("="*50)
        print(f"ERROR: '{VOCAB_FILE}' not found or is empty.")
        print("Please create it and add words, one per line.")
        print("Using fallback vocabulary.")
        print("="*50)
        VOCABULARY = [
            "Harbor", "Signal", "Forest", "Circuit", "Embassy", "Runway",
            "Gallery", "Cipher", "Orbit", "Station", "Contract", "Anchor",
            "Customs", "Voltage", "Algorithm", "Monument", "Transit",
            "Mercury", "Market", "Archive", "District", "Theater", "Summit",
            "Frame", "Delta", "Bridge", "Field", "Carrier", "Memory",
            "Chain", "Shell", "Module", "Crown", "Line", "Vault"
        ]
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        pass

# --- 3. Gemini LLM Configuration ---

# <<< MODIFIED: This section is now for *all* LLM clients and prompts >>>

# --- Prompt Templates ---
# We keep the LLM-specific instruction templates separate.
GAME_INSTRUCTIONS = """
You are the ranking AI for a word association game.
The user will provide a "Clue" and a "List of Words".
Your job is to rank all words in the list by their semantic association to the clue, from MOST related to LEAST related.
Respond with ONLY the ranked list of words, separated by newlines.
Do not add any explanation, headers, or other text.
"""

GEMINI_PROMPT_TEMPLATE = """
{instructions}

Clue: {clue}
List of Words:
{word_list}

Ranked List:
"""

OPENAI_USER_PROMPT_TEMPLATE = """
Clue: {clue}
List of Words:
{word_list}

Ranked List:
"""

# --- Generation Configs ---
gemini_generation_config = {
    "temperature": 0.0,
    "max_output_tokens": 512,
}

# --- Initialize LLM Clients ---
gemini_model = None
openai_client = None

if LLM_PROVIDER == 'gemini':
    try:
        if not GEMINI_API_KEY:
            raise KeyError("GEMINI_API_KEY not set in .env file")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite", # Using latest flash model
            generation_config=gemini_generation_config
        )
        print("--- LLM Provider: Gemini ---")
    except Exception as e:
        print(f"Warning: Could not initialize Gemini model. /rank will fail. Error: {e}")

elif LLM_PROVIDER == 'openai':
    try:
        if not OPENAI_API_KEY:
            raise KeyError("OPENAI_API_KEY not set in .env file")
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Test credentials with a simple model list
        openai_client.models.list()
        print(f"--- LLM Provider: OpenAI (Model: {OPENAI_MODEL_NAME}) ---")
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client. /rank will fail. Error: {e}")
        openai_client = None # Ensure it's None on failure
elif LLM_PROVIDER == 'custom':
    try:
        if not CUSTOM_API_KEY:
            raise KeyError("CUSTOM_API_KEY not set in .env file")
        if not CUSTOM_ENDPOINT_URL:
            raise KeyError("CUSTOM_ENDPOINT_URL not set in .env file")

        openai_client = openai.OpenAI(
            api_key=CUSTOM_API_KEY,
            base_url=CUSTOM_ENDPOINT_URL
        )

        print(f"--- LLM Provider: Custom (SiliconFlow) ---")
        print(f"--- Model: {CUSTOM_MODEL_NAME} ---")
        print(f"--- Endpoint: {CUSTOM_ENDPOINT_URL} ---")

    except Exception as e:
        print(f"Warning: Could not initialize Custom LLM client. /rank will fail. Error: {e}")
        openai_client = None

else:
    print(f"Warning: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. /rank will fail.")


# --- 4. Helper Functions ---

def get_new_words(seen_words, num_to_add):
    """
    Gets new random words from VOCABULARY, ensuring they have not been 'seen'.
    """
    seen_set = set(w.lower() for w in seen_words)
    available_vocab = [w for w in VOCABULARY if w.lower() not in seen_set]
    
    num_possible = len(available_vocab)

    if num_possible == 0:
        return [] 

    num_to_get = min(num_possible, num_to_add)
    
    if num_to_get == 0:
        return []
        
    new_words = random.sample(available_vocab, num_to_get)
    return new_words

def parse_ranked_list(response_text, current_board):
    """
    Parses the LLM's newline-separated list and validates it.
    Returns the ranked list or None if validation fails.
    """
    if not response_text:
        print("LLM Validation Error: Received empty response.")
        return None

    ranked_words = [word.strip() for word in response_text.strip().split('\n')]
    
    # Simple filter to remove empty strings that might result from LLM
    ranked_words = [w for w in ranked_words if w]
    
    if sorted(w.lower() for w in ranked_words) == sorted(w.lower() for w in current_board):
        return ranked_words
    else:
        print("LLM Validation Error:")
        print(f"  Expected: {sorted([w.lower() for w in current_board])}")
        print(f"  Got: {sorted([w.lower() for w in ranked_words])}")
        return None


# <<< NEW: LLM Abstraction Function >>>
def get_llm_ranking(clue, word_list_str):
    """
    Calls the configured LLM provider and returns the raw response text.
    This function is designed to "fail fast" and will raise an
    exception if the API call fails, which will be caught by the /rank endpoint.
    """
    global LLM_PROVIDER, gemini_model, openai_client

    if LLM_PROVIDER == 'gemini':
        if not gemini_model:
            raise Exception("Gemini model not initialized.")
        
        prompt = GEMINI_PROMPT_TEMPLATE.format(
            instructions=GAME_INSTRUCTIONS,
            clue=clue,
            word_list=word_list_str
        )
        response = gemini_model.generate_content(prompt)
        return response.text

    elif LLM_PROVIDER == 'openai':
        if not openai_client:
            raise Exception("OpenAI client not initialized.")
        
        user_prompt = OPENAI_USER_PROMPT_TEMPLATE.format(
            clue=clue,
            word_list=word_list_str
        )
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": GAME_INSTRUCTIONS},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    elif LLM_PROVIDER == 'custom':
            if not openai_client:
                raise Exception("OpenAI-Custom client not initialized.")
            
            user_prompt = OPENAI_USER_PROMPT_TEMPLATE.format(
                clue=clue,
                word_list=word_list_str
            )
            response = openai_client.chat.completions.create(
                model=CUSTOM_MODEL_NAME,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": GAME_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content

    else:
        raise Exception(f"No valid LLM_PROVIDER configured ('{LLM_PROVIDER}').")


# --- 5. Flask API Endpoints ---

@app.route('/')
def index():
    """
    Serves the main game page.
    Initializes the game state (score, board, target, seen_words).
    """
    session.clear()
    
    session['score'] = 0
    session['seen_words'] = [] 

    initial_size = calculate_board_size(0)
    initial_board = get_new_words(session['seen_words'], initial_size)

    if not initial_board:
        # Handle edge case where vocabulary is smaller than initial size
        print("CRITICAL: Not enough words in vocabulary to start game.")
        return "Error: Vocabulary is too small to start the game.", 500

    session['board'] = initial_board
    session['target_word'] = random.choice(initial_board)
    session['seen_words'] = list(initial_board)
    session['game_over'] = False

    return render_template(
        'arcade.html',
        game_board=session['board'],
        target_word=session['target_word'],
        score=session['score']
    )


@app.route('/rank', methods=['POST'])
def rank_words():
    """
    This is the core "LLM Ranking Engine" API for Arcade Mode.
    Takes a clue, gets a ranked list, checks for target, and updates score/board.
    Implements persistent re-ranking and a win condition.
    """
    # --- 1. Check for Win Condition (Game Over) ---
    if session.get('game_over', False):
        return jsonify({
            'hit': False, 
            'game_over': True, 
            'new_target': "YOU WIN!",
            'new_board': session.get('board', []),
            'new_score': session.get('score', 0),
            'error': 'Game is over. Please refresh to start a new game.'
        }), 400

    # <<< MODIFIED: Removed 'if not model' check, handled by abstraction >>>

    data = request.json
    clue = data.get('clue')
    if not clue:
        return jsonify({'error': 'No clue provided'}), 400

    # --- 2. Load Session State ---
    current_board = session.get('board', [])
    target_word = session.get('target_word', '')
    score = session.get('score', 0)
    seen_words = session.get('seen_words', [])
    
    if not current_board or not target_word:
        return jsonify({'error': 'Game session error. Please refresh.'}), 400

    # --- 3. Call LLM Abstraction ---
    # <<< MODIFIED: This block now calls our abstraction function >>>
    start_time = time.perf_counter()
    word_list_str = "\n".join(current_board)
    
    try:
        # This one function call now handles Gemini, OpenAI, or Custom
        response_text = get_llm_ranking(clue, word_list_str)
        
        ranked_list = parse_ranked_list(response_text, current_board)
        
        if not ranked_list:
            # This handles both empty responses and validation failures
            raise Exception("AI response was invalid or didn't match the board.")
        
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000)

    except Exception as e:
        # This 'except' block gracefully handles errors from *any* provider
        # (e.g., API key error, network error, parsing error, 503 status)
        print(f"WARNING: LLM API or parsing error: {e}")
        
        # "Continue as normal": Return the *current* state without
        # changes, signaling a "miss" with a special error flag.
        return jsonify({
            'hit': False,
            'game_over': False,
            'words_removed': [],
            'new_board': current_board,      # Send the board *as it was*
            'new_target': target_word,       # Send the target *as it was*
            'new_score': score,              # Send the score *as it was*
            'ranked_list': current_board,    # For consistency, send the un-ranked board
            'latency_ms': 0,
            'api_error': True, # Flag for the frontend
            'error_message': f'API error: {str(e)}'
        })
    # <<< END MODIFIED BLOCK >>>

    # --- 4. NEW MECHANIC: Persistent Re-ranking ---
    # This code only runs if the 'try' block succeeded
    session['board'] = ranked_list

    # --- 5. Find Target and Check for Hit ---
    try:
        target_index = next(i for i, w in enumerate(ranked_list) if w.lower() == target_word.lower())
    except StopIteration:
        # This should be rare, but handle it as a graceful error
        print(f"CRITICAL ERROR: Target word '{target_word}' missing from validated list.")
        return jsonify({'error': f'Target word "{target_word}" not found in AI response.'}), 500

    # Check if the target is in the bottom 4 (index 0, 1, 2, or 3)
    if 0 <= target_index <= 3:
        # --- HIT! ---
        num_to_remove = 4 - target_index
        start = target_index
        end = start + num_to_remove
        words_removed = ranked_list[start:end]
        
        score += len(words_removed)
        
        words_to_keep = [w for w in ranked_list if w not in words_removed]
        
        desired_size = calculate_board_size(score)
        missing = max(0, desired_size - len(words_to_keep))

        # --- 6. NEW MECHANIC: Get New Words & Check Win Condition ---
        new_words = get_new_words(seen_words, missing)

        if not new_words and missing > 0:
            # WIN CONDITION: We needed words, but the vocabulary is empty.
            session['game_over'] = True
            session['board'] = words_to_keep
            session['score'] = score
            
            return jsonify({
                'hit': True,
                'game_over': True,
                'words_removed': words_removed,
                'new_board': words_to_keep,
                'new_target': "YOU WIN!",
                'new_score': score,
                'ranked_list': ranked_list,
                'latency_ms': latency_ms
            })
        
        # --- 7. (HIT) Update Board & Session ---
        new_board = words_to_keep + new_words
        
        # Handle rare case where board is empty after removal but before win
        if not new_board:
             return jsonify({'error': 'Game error: Board became empty.'}), 500

        new_target = random.choice(new_words) if new_words else random.choice(new_board)

        session['board'] = new_board
        session['score'] = score
        session['target_word'] = new_target
        session['seen_words'] = seen_words + new_words

        return jsonify({
            'hit': True,
            'game_over': False,
            'words_removed': words_removed,
            'new_board': new_board,
            'new_target': new_target,
            'new_score': score,
            'ranked_list': ranked_list, 
            'latency_ms': latency_ms
        })

    else:
        # --- MISS! ---
        return jsonify({
            'hit': False,
            'game_over': False,
            'words_removed': [],
            'new_board': ranked_list,
            'new_target': target_word, 
            'new_score': score, 
            'ranked_list': ranked_list,
            'latency_ms': latency_ms,
            'api_error': False # Explicitly show no error
        })

# --- 6. Run the Application ---

if __name__ == '__main__':
    load_vocabulary() 
    if not VOCABULARY:
        print("CRITICAL: No vocabulary loaded. Exiting.")
    else:
        # We print the provider *after* the client initialization attempts
        print(f"--- Starting Flask app. (http://127.0.0.1:5001) ---")
        app.run(debug=True, port=5001)
import os
import time
import random
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

# Load environment variables from .env file (for GEMINI_API_KEY, FLASK_SECRET_KEY)
load_dotenv()

# --- 1. Configuration ---

app = Flask(__name__)
# SECRET_KEY is required for Flask sessions.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Configure Gemini API
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("="*50)
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set this variable in your .env file or environment.")
    print("="*50)
    pass

# --- 2. Game Content ---
def calculate_board_size(score):
    """
    Minimum 5, maximum 20.
    Board size = min(20, max(5, score//2))
    """
    size = max(5, score // 2)
    return min(size, 22)
VOCABULARY = []
VOCAB_FILE = os.path.join('assets', 'aviation_1.txt')

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

GEMINI_PROMPT_TEMPLATE = """
You are the ranking AI for a word association game.
The user will provide a "Clue" and a "List of Words".
Your job is to rank all words in the list by their semantic association to the clue, from MOST related to LEAST related.
Respond with ONLY the ranked list of words, separated by newlines.
Do not add any explanation, headers, or other text.

Clue: {clue}
List of Words:
{word_list}

Ranked List:
"""

generation_config = {
    "temperature": 0.0,
    "max_output_tokens": 512, 
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite-preview-09-2025", # Using latest flash model
        generation_config=generation_config
    )
except Exception as e:
    print(f"Warning: Could not initialize Gemini model. /rank will fail. Error: {e}")
    model = None

# --- 4. Helper Functions ---

def get_new_words(current_board, num_to_add):
    """Gets new random words, ensuring no duplicates."""
    available_vocab = [w for w in VOCABULARY if w not in current_board]
    num_possible = len(available_vocab)
    
    if num_possible < num_to_add:
        new_words = random.choices(VOCABULARY, k=num_to_add)
    else:
        new_words = random.sample(available_vocab, num_to_add)
        
    return new_words

def parse_ranked_list(response_text, current_board):
    """
    Parses the LLM's newline-separated list and validates it.
    Returns the ranked list or None if validation fails.
    """
    ranked_words = [word.strip() for word in response_text.strip().split('\n')]
    
    if sorted(w.lower() for w in ranked_words) == sorted(w.lower() for w in current_board):
        return ranked_words
    else:
        print("LLM Validation Error:")
        print(f"  Expected: {sorted(current_board)}")
        print(f"  Got: {sorted(ranked_words)}")
        return None


# --- 5. Flask API Endpoints ---

@app.route('/')
def index():
    """
    Serves the main game page.
    Initializes the game state (score, board, target).
    """
    session.clear()  # Start a new game
    
    # Initialize score
    session['score'] = 0

    # Dynamic board size using the scoring function
    initial_size = calculate_board_size(0)     # score = 0 → minimum 5
    initial_board = get_new_words([], initial_size)

    session['board'] = initial_board
    session['target_word'] = random.choice(initial_board)

    return render_template(
        'arcade.html',
        game_board=session['board'],
        target_word=session['target_word'],
        score=session['score']
    )

# -----------------------------------------------------------------
# NOTE: The duplicated index() route that was here has been removed.
# -----------------------------------------------------------------


@app.route('/rank', methods=['POST'])
def rank_words():
    """
    This is the core "LLM Ranking Engine" API for Arcade Mode.
    Takes a clue, gets a ranked list, checks for target, and updates score/board.
    """
    if not model:
        return jsonify({'error': 'Gemini model not initialized. Is API key set?'}), 500

    data = request.json
    clue = data.get('clue')
    if not clue:
        return jsonify({'error': 'No clue provided'}), 400

    current_board = session.get('board', [])
    target_word = session.get('target_word', '')
    score = session.get('score', 0)
    
    if not current_board or not target_word:
        return jsonify({'error': 'Game session error. Please refresh.'}), 400

    # --- Call Gemini API ---
    start_time = time.perf_counter()
    word_list_str = "\n".join(current_board)
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        clue=clue,
        word_list=word_list_str
    )
    
    try:
        response = model.generate_content(prompt)
        ranked_list = parse_ranked_list(response.text, current_board)
        
        if not ranked_list:
            raise Exception("AI response was invalid or didn't match the board.")

    except Exception as e:
        print(f"Gemini API error: {e}")
        return jsonify({'error': f'Gemini API error: {str(e)}'}), 500

    end_time = time.perf_counter()
    latency_ms = round((end_time - start_time) * 1000)

    # --- Arcade Game Logic ---
    try:
        target_index = next(i for i, w in enumerate(ranked_list) if w.lower() == target_word.lower())
    except StopIteration:
        return jsonify({'error': f'Target word "{target_word}" not found in AI response.'}), 500

    # Check if the target is in the bottom 4 (index 0, 1, 2, or 3)
    # This logic matches your description:
    # index 0 (most correlated) -> removes 4
    # index 3 (4th correlated) -> removes 1
    if 0 <= target_index <= 3:
        # --- HIT! ---
        # We want to remove the target *and* all words between it and the 4th-most related.
        # ranked_list is sorted MOST related (index 0) -> LEAST related
        # So we remove indices [target_index .. 3] inclusive.
        num_to_remove = 4 - target_index          # 4, 3, 2, or 1
        start = target_index                      # ensure target is always removed
        end = start + num_to_remove               # slice end (exclusive) => up to index 3
        words_removed = ranked_list[start:end]    # e.g. idx 2 -> remove [2,3]
        #words
        
        score += len(words_removed)
        
        words_to_keep = [w for w in current_board if w not in words_removed]
        desired_size = calculate_board_size(score)
        missing = max(0, desired_size - len(words_to_keep))

        new_words = get_new_words(words_to_keep, missing)
        new_board = words_to_keep + new_words
        
        new_target = random.choice(new_words) if new_words else random.choice(new_board)

        session['board'] = new_board
        session['score'] = score
        session['target_word'] = new_target

        return jsonify({
            'hit': True,
            'words_removed': words_removed,
            'new_board': new_board,
            'new_target': new_target,
            'new_score': score,
            'ranked_list': ranked_list, 
            'latency_ms': latency_ms
        })

    else:
        # --- MISS! ---
        session['board'] = ranked_list

        return jsonify({
            'hit': False,
            'words_removed': [],
            'new_board': ranked_list, 
            'new_target': target_word, 
            'new_score': score, 
            'ranked_list': ranked_list,
            'latency_ms': latency_ms
        })

# --- 6. Run the Application ---

if __name__ == '__main__':
    load_vocabulary() 
    if not VOCABULARY:
        print("CRITICAL: No vocabulary loaded. Exiting.")
    else:
        app.run(debug=True, port=5001)
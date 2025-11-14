# **LLM-Powered Semantris Tower — A Modern Reimagining**

This project is a small, experimental re-creation of the brilliant **Semantris** concept.  
The original Semantris was—and still is—an amazing linguistic puzzle built around clever word association. It helped people *learn wording, unlock creativity, spark intuition,* and generate surprising semantic connections.

But it was also a product of its time.

Since the introduction of **ChatGPT-3.5 in late 2022**, the entire world has shifted.  
AI LLMs have exploded in capability, scale, reasoning power, and sheer cultural impact.  
With modern models carrying **mountains of knowledge across every domain**, it felt natural to revisit Semantris through the lens of 2025’s LLM era.

And so, this project was born — a **lightweight LLM-powered arcade variant** that uses the spirit of Semantris but updates it with modern AI.



---

## **🧠 What This Project Does**
This game uses **Google’s Gemini** LLM to:

- Rank words by semantic association  
- Determine hits vs misses  
- Dynamically reshuffle a tower of words  
- Reward clever or accurate clues  
- Enable truly modern semantic reasoning that stays up-to-date with today’s world

It is *not* deterministic (even at temperature 0), because LLMs aren’t strict sorting machines.  
However, for short phrases and word association, the result is **surprisingly consistent and fun**.

The implementation includes a small sample vocabulary stored as `.txt` files.  
Feel free to replace them with **any of your own word lists**:

- Search for `.txt` in `app.py`
- Replace with another file in the `assets/` folder
- Enjoy instant new themes (aviation, cities, sci-fi, geography, etc.)



---

## **📦 Setup Instructions**

1. Install dependencies:

   Use the included `requirements.txt`.

2. Create a `.env` file in the project root with:

GEMINI_API_KEY="YOUR_API_KEY"
FLASK_SECRET_KEY="YOUR_SECRET_KEY"


A free-tier API key from **Google AI Studio** is usually enough for this demo.



---

## **🎮 Current State of the Game**

This is a **quick afternoon prototype**, not a polished product.  
Here are the known issues and limitations:

### ✔ Core mechanics  
LLM ranking, scoring, tower logic, and gameplay loop all work.

### ✖ Animation  
Still under development.  
FLIP animations work but could be smoother.

### ✖ UI / UX  
Functional but not pretty — definitely “engineer art.”  
Future PRs for styling or redesign are very welcome.

### ✖ Ending Condition  
There is currently **no real end**.  
In the short term, the intended “finish line” is:

**Who completes the same vocabulary set with the shortest time and highest score.**

Timer + score are included for this reason.



---

## **🤝 Contributing**

**Pull requests are welcome across every front:**

- UI improvements  
- Animations  
- Better game balancing  
- Theme packs / vocab files  
- End-game logic  
- Performance optimizations  
- LLM-prompt enhancement  
- Anything creative!

This is just a small demo; contributions can elevate it enormously.



---

## **🙏 Acknowledgements**

A multitude of modern AI LLMs inspired this.  
Gemini powers the engine today, but respect goes to:

- OpenAI  
- Google  
- Anthropic  
- DeepSeek
- Meta  
- Qwen  
- And many more

These models have become **force multipliers of the decade**,  
reshaping how developers, creators, and everyday people build things.

This project is nowhere close to the groundbreaking technical architecture or UI polish of Google’s original Semantris — that was built by an amazing team of engineers across many stacks.  
This is just a humble modern twist using tools available to everyone.



---

## **Enjoy the Game! 🎉**

Experiment with word lists.  
Play with clues.  
Try creative associations.  
Push the limits of LLM semantics.  

And most importantly:

**Have fun exploring how modern AI interprets the world through words.**

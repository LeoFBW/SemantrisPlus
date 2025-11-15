import os
import time
import httpx  # Using httpx for async mock requests
import statistics
import asyncio  # Import asyncio
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Import Gemini + OpenAI clients
import google.generativeai as genai
from openai import AsyncOpenAI  # Import AsyncOpenAI


# ============================================================
#                CONFIGURATION
# ============================================================

# NOTE: The script now ignores API_PROVIDER and tests the list below.
NUM_REQUESTS_PER_MODEL = 5
CLUE = "Watercraft"
WORDS = ["Harbor", "Signal", "Forest", "Circuit", "Embassy",
         "Runway", "Gallery", "Cipher", "Orbit", "Station",
         "Contract", "Anchor"]

MOCK_URL = "https://httpbin.org/post"

PROMPT_TEMPLATE = """
Rank all words in the list by their semantic association to the clue.
Return ONLY the single best (most related) word.

Clue: {clue}
Words: {words}
"""

# -----------------------------------------------------------------
#    MODELS TO TEST (Curated from your provided list)
# -----------------------------------------------------------------
# I've selected a variety of text-based models.
# You can add/remove any model ID from the full list below.

RECOMMENDED_MODELS_TO_TEST = [
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "moonshotai/Kimi-K2-Thinking",
    "MiniMaxAI/MiniMax-M2",
    "zai-org/GLM-4.6",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "inclusionAI/Ring-flash-2.0",
    "inclusionAI/Ling-flash-2.0",
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
    "tencent/Hunyuan-A13B-Instruct",
    "moonshotai/Kimi-Dev-72B",
    "deepseek-ai/DeepSeek-V2.5",
]

# --- FULL FILTERED TEXT-ONLY MODEL LIST ---
# If you want to run a very long and comprehensive test,
# you can replace the list above with this one.
# (I have filtered out all Vision/-VL, -OCR, -Captioner,
# and free/pro duplicates from your original paste)

FULL_TEXT_MODEL_LIST = [
    # "deepseek-ai/DeepSeek-V3.1-Terminus",
    # "deepseek-ai/DeepSeek-V3.2-Exp",
    # "deepseek-ai/DeepSeek-R1",
    # "deepseek-ai/DeepSeek-V3",
    # "moonshotai/Kimi-K2-Thinking",
    # "MiniMaxAI/MiniMax-M2",
    # "inclusionAI/Ring-1T",
    # "inclusionAI/Ling-1T",
    # "zai-org/GLM-4.6",
    # "Kwaipilot/KAT-Dev",
    # "moonshotai/Kimi-K2-Instruct-0905",
    # "Qwen/Qwen3-Next-80B-A3B-Instruct",
    # "Qwen/Qwen3-Next-80B-A3B-Thinking",
    # "inclusionAI/Ring-flash-2.0",
    # "inclusionAI/Ling-flash-2.0",
    # "inclusionAI/Ling-mini-2.0",
    # "ByteDance-Seed/Seed-OSS-36B-Instruct",
    # "zai-org/GLM-4.5",
    # "zai-org/GLM-4.5-Air",
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    # "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # "Qwen/Qwen3-30B-A3B-Instruct-2507",
    # "Qwen/Qwen3-235B-A22B-Thinking-2507",
    # "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # "baidu/ERNIE-4.5-300B-A47B",
    # "tencent/Hunyuan-A13B-Instruct",
    # "moonshotai/Kimi-Dev-72B",
    # "MiniMaxAI/MiniMax-M1-80k",
    # "Tongyi-Zhiwen/QwenLong-L1-32B",
    # "Qwen/Qwen3-30B-A3B",
    # "Qwen/Qwen3-32B",
    # "Qwen/Qwen3-14B",
    # "ascend-tribe/pangu-pro-moe",
    # "THUDM/GLM-Z1-32B-0414",
    # "THUDM/GLM-4-32B-0414",
    # "Qwen/Qwen3-235B-A22B",
    # "Qwen/QwQ-32B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # "deepseek-ai/DeepSeek-V2.5",
    # "Qwen/Qwen2.5-Coder-32B-Instruct",
    # "Qwen/Qwen2.5-72B-Instruct-128K",
    # "Qwen/Qwen2.5-72B-Instruct",
    # "Qwen/Qwen2.5-32B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "THUDM/GLM-Z1-Rumination-32B-0414",
]


# ============================================================
#             LLM CLIENT ABSTRACTIONS
# ============================================================

class LLMClientBase:
    """Base interface for all providers."""
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def close(self):
        """Allow clients (like httpx) to clean up."""
        pass


# -------------------- MOCK PROVIDER -------------------------

class MockClient(LLMClientBase):
    def __init__(self):
        # Use httpx.AsyncClient for async requests
        self.client = httpx.AsyncClient(timeout=15)

    async def generate(self, prompt: str) -> str:
        """
        We do NOT call LLM here, instead we only call httpbin to measure pure network latency.
        """
        response = await self.client.post(MOCK_URL, json={"prompt": prompt})
        response.raise_for_status()
        return "mock-result"

    async def close(self):
        await self.client.aclose()


# -------------------- GEMINI PROVIDER ------------------------

class GeminiClient(LLMClientBase):
    def __init__(self, model="gemini-2.5-flash-lite-preview-09-2025"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 20
            }
        )

    async def generate(self, prompt: str) -> str:
        # Use the async method
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()


# -------------------- OPENAI (SILICONFLOW) PROVIDER ------------------------

class OpenAIClient(LLMClientBase):
    def __init__(self, model="deepseek-ai/DeepSeek-V3.2-Exp"):
        api_key = os.getenv("SiliconFlow_API_KEY")
        if not api_key:
            raise ValueError("SiliconFlow_API_KEY not set in environment")

        # Use AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.model = model

    async def generate(self, prompt: str) -> str:
        # Use 'await' for the async call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()


# ============================================================
#               LATENCY TESTING ENGINE
# ============================================================

async def test_model_performance(model_name: str, num: int, prompt: str):
    """
    Tests a single model by running 'num' requests concurrently.
    """
    print(f"\n--- Testing Model: {model_name} ({num} requests) ---")
    
    # Always use OpenAIClient for this benchmark
    client = OpenAIClient(model=model_name)
    
    latencies = []
    results = []
    
    # Create a list of tasks to run concurrently
    tasks = [client.generate(prompt) for _ in range(num)]
    
    start_time = time.perf_counter()
    
    # Run all tasks concurrently
    # return_exceptions=True so one failure doesn't stop the batch
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_duration_s = time.perf_counter() - start_time
    
    num_success = 0
    num_fail = 0
    
    for res in task_results:
        if isinstance(res, Exception):
            print(f"[!] Request FAILED: {res}")
            num_fail += 1
        else:
            # We can't get individual latencies easily this way,
            # but we get the total time, which is what matters for a batch.
            num_success += 1
            results.append(res)
            
    print(f"--> Completed in: {total_duration_s:.2f} s")
    print(f"--> Success/Fail: {num_success} / {num_fail}")
    
    # Calculate stats for individual requests if needed (less accurate)
    avg_latency_ms = (total_duration_s * 1000) / num if num > 0 else 0
    
    if results:
        # Check if all results are the same (consistency)
        first_result = results[0]
        if all(r == first_result for r in results):
            print(f"--> Consistent Result: '{first_result}'")
        else:
            print(f"--> INCONSISTENT Results: {set(results)}")
            
    await client.close() # Clean up client if needed

    return {
        "model_name": model_name,
        "total_time_s": total_duration_s,
        "num_success": num_success,
        "num_fail": num_fail,
        "avg_latency_ms_approx": avg_latency_ms
    }


def print_summary_report(all_stats: list):
    if not all_stats:
        print("\nNo models were tested.")
        return

    print("\n\n" + "="*50)
    print("         PERFORMANCE BENCHMARK SUMMARY")
    print("="*50)

    # Sort by total time, fastest first
    sorted_stats = sorted(all_stats, key=lambda x: x["total_time_s"])

    print(f"\nTested {len(sorted_stats)} models ({NUM_REQUESTS_PER_MODEL} concurrent requests each)")
    print("\n--- Results (Sorted by Fastest Total Time) ---")
    
    # Header
    print(f"{'Rank':<5} | {'Model Name':<40} | {'Total Time':<12} | {'Success':<10} | {'Avg (ms)':<10}")
    print(f"{'-'*5:<5} | {'-'*40:<40} | {'-'*12:<12} | {'-'*10:<10} | {'-'*10:<10}")

    for i, stats in enumerate(sorted_stats):
        rank = f"{i+1}."
        model = stats['model_name']
        total_t = f"{stats['total_time_s']:.2f} s"
        success = f"{stats['num_success']}/{stats['num_success'] + stats['num_fail']}"
        avg_ms = f"{stats['avg_latency_ms_approx']:.0f}"

        print(f"{rank:<5} | {model:<40} | {total_t:<12} | {success:<10} | {avg_ms:<10}")


# ============================================================
#                      MAIN LOGIC
# ============================================================

async def main():
    """
    Main async function to run the benchmark.
    """
    prompt = PROMPT_TEMPLATE.format(clue=CLUE, words=str(WORDS))
    all_model_stats = []
    
    models_to_run = RECOMMENDED_MODELS_TO_TEST
    
    print(f"Starting benchmark for {len(models_to_run)} models...")

    for model_name in models_to_run:
        try:
            stats = await test_model_performance(
                model_name=model_name,
                num=NUM_REQUESTS_PER_MODEL,
                prompt=prompt
            )
            all_model_stats.append(stats)
        except Exception as e:
            print(f"\n[!!!] CRITICAL FAILURE testing {model_name}: {e}")
            all_model_stats.append({
                "model_name": model_name,
                "total_time_s": float('inf'), # Sorts to the bottom
                "num_success": 0,
                "num_fail": NUM_REQUESTS_PER_MODEL,
                "avg_latency_ms_approx": float('inf')
            })
        
        await asyncio.sleep(0.5) # Small buffer between models

    print_summary_report(all_model_stats)


if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
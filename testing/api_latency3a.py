import os
import time
import requests
import statistics
import asyncio
import aiohttp  # For async mock client 
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Import Gemini + OpenAI clients
import google.generativeai as genai
from openai import AsyncOpenAI  # Import the Async client


# ============================================================
#                CONFIGURATION
# ============================================================

# --- Your new test parameters ---
NUM_REQUESTS = 5       # Number of tries per model
REQUEST_TIMEOUT = 10.0 # Max seconds per request

# --- Models to Test ---
# We will test this one Gemini model
GEMINI_MODELS = [
    "gemini-2.5-flash-lite-preview-09-2025"
]

# ...and all of these SiliconFlow models
SILICONFLOW_MODELS = [
    "deepseek-ai/DeepSeek-V3.1-Terminus", "deepseek-ai/DeepSeek-V3.2-Exp",
    "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3",
    "moonshotai/Kimi-K2-Thinking", "MiniMaxAI/MiniMax-M2",
    "inclusionAI/Ring-1T", "inclusionAI/Ling-1T", "zai-org/GLM-4.6",
    "Kwaipilot/KAT-Dev", "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking", "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking", "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking", "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Thinking", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking", "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "deepseek-ai/DeepSeek-OCR", "moonshotai/Kimi-K2-Instruct-0905",
    "Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "inclusionAI/Ring-flash-2.0", "inclusionAI/Ling-flash-2.0",
    "inclusionAI/Ling-mini-2.0", "tencent/Hunyuan-MT-7B",
    "ByteDance-Seed/Seed-OSS-36B-Instruct", "zai-org/GLM-4.5V",
    "zai-org/GLM-4.5", "zai-org/GLM-4.5-Air", "stepfun-ai/step3",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "THUDM/GLM-4.1V-9B-Thinking", "Pro/THUDM/GLM-4.1V-9B-Thinking",
    "baidu/ERNIE-4.5-300B-A47B", "tencent/Hunyuan-A13B-Instruct",
    "moonshotai/Kimi-Dev-72B", "MiniMaxAI/MiniMax-M1-80k",
    "Tongyi-Zhiwen/QwenLong-L1-32B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B", "Qwen/Qwen3-14B",
    "ascend-tribe/pangu-pro-moe", "THUDM/GLM-Z1-32B-0414",
    "THUDM/GLM-4-32B-0414", "THUDM/GLM-Z1-9B-0414", "THUDM/GLM-4-9B-0414",
    "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen3-235B-A22B", "Qwen/QwQ-32B",
    "Qwen/Qwen2.5-VL-72B-Instruct", "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Qwen/QVQ-72B-Preview",
    "deepseek-ai/DeepSeek-V2.5", "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct-128K",
    "deepseek-ai/deepseek-vl2", "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct",
    "internlm/internlm2_5-7b-chat", "Qwen/Qwen2-7B-Instruct",
    "THUDM/glm-4-9b-chat", "Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Pro/Qwen/Qwen2.5-Coder-7B-Instruct", "Pro/Qwen/Qwen2.5-7B-Instruct",
    "Pro/Qwen/Qwen2-7B-Instruct", "Pro/THUDM/glm-4-9b-chat",
    "THUDM/GLM-Z1-Rumination-32B-0414"
]


# --- Task Configuration ---
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


# ============================================================
#             LLM CLIENT ABSTRACTIONS
# ============================================================

class LLMClientBase:
    """Base interface for all providers."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def close(self):
        """Hook for cleanup, e.g., closing http sessions."""
        pass


# -------------------- MOCK PROVIDER -------------------------

class MockClient(LLMClientBase):
    def __init__(self, model_name="mock-model"):
        super().__init__(model_name)
        # Create the session at init
        self.session = aiohttp.ClientSession()

    async def generate(self, prompt: str) -> str:
        """
        We do NOT call LLM here, instead we only call httpbin to measure pure network latency.
        """
        async with self.session.post(MOCK_URL, json={"prompt": prompt}) as response:
            response.raise_for_status()
            return "mock-result"

    async def close(self):
        """Close the persistent session."""
        await self.session.close()


# -------------------- GEMINI PROVIDER ------------------------

class GeminiClient(LLMClientBase):
    def __init__(self, model_name="gemini-2.5-flash-lite-preview-09-2025"):
        super().__init__(model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 20
            }
        )

    async def generate(self, prompt: str) -> str:
        # Use the async version of the call
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()


# -------------------- OPENAI (SILICONFLOW) PROVIDER ------------------------

class OpenAIClient(LLMClientBase):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        api_key = os.getenv("SiliconFlow_API_KEY")
        if not api_key:
            raise ValueError("SiliconFlow_API_KEY not set in environment")

        # Use the AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )

    async def generate(self, prompt: str) -> str:
        # Use 'await' for the async call
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()


# ============================================================
#               LATENCY TESTING ENGINE
# ============================================================

def calculate_stats(provider: str, model: str, latencies: list, failures: int, timeouts: int) -> dict:
    """Helper function to compile stats for one model."""
    successes = len(latencies)
    
    return {
        "provider": provider,
        "model": model,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "avg_ms": statistics.mean(latencies) if latencies else 0,
        "median_ms": statistics.median(latencies) if latencies else 0,
        "min_ms": min(latencies) if latencies else 0,
        "max_ms": max(latencies) if latencies else 0,
    }


async def run_model_test(
    client: LLMClientBase, 
    provider_name: str, 
    prompt: str, 
    num: int, 
    timeout: float
) -> dict:
    """
    Runs a full latency test (N requests) for a *single* model.
    """
    latencies = []
    failures = 0
    timeouts = 0
    model_name = client.model_name

    print(f"[STARTING] Testing {provider_name} model: {model_name} ({num} requests)...")

    for i in range(num):
        start = time.perf_counter()
        try:
            # Run the client's generate() method with a specific timeout
            result = await asyncio.wait_for(
                client.generate(prompt), 
                timeout=timeout
            )
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            latencies.append(duration_ms)
            
            # Short-hand print
            print(f"  > {model_name} ({i+1}/{num}): Success {duration_ms:.0f} ms")

        except asyncio.TimeoutError:
            timeouts += 1
            print(f"  > {model_name} ({i+1}/{num}): FAILED (Timeout > {timeout}s)")
        
        except Exception as e:
            failures += 1
            # Print a snippet of the error
            error_str = str(e).split('\n')[0]
            print(f"  > {model_name} ({i+1}/{num}): FAILED ({error_str[:70]}...)")

        # Sleep to avoid basic rate-limiting
        await asyncio.sleep(0.5) 

    print(f"[FINISHED] {model_name}")
    
    # Clean up the client (e.g., close http session)
    await client.close()
    
    # Return a structured dictionary of the results
    return calculate_stats(provider_name, model_name, latencies, failures, timeouts)


def print_summary_report(results: list):
    """
    Prints a formatted markdown table of all results, sorted by best avg latency.
    """
    
    # Sort by average latency (fastest first). 
    # Put models with 0 successes (infinite latency) at the end.
    sorted_results = sorted(
        results, 
        key=lambda x: (x['avg_ms'] if x['successes'] > 0 else float('inf'))
    )

    print("\n\n--- 🏁 Final Performance Report 🏁 ---")
    print("\nSorted by fastest average latency (lower is better).\n")
    
    # Print Header
    print(
        f"| {'Rank':<4} | {'Model':<45} | {'Provider':<12} | {'Avg (ms)':<10} | {'Median (ms)':<11} | {'Success':<7} | {'Failed':<6} | {'Timeout':<7} |"
    )
    print(
        f"|{'-'*6:}|{'-'*47:}|{'-'*14:}|{'-'*12:}|{'-'*13:}|{'-'*9:}|{'-'*8:}|{'-'*9:}|"
    )

    # Print Rows
    for i, res in enumerate(sorted_results):
        print(
            f"| {i+1:<4} | {res['model']:<45} | {res['provider']:<12} | "
            f"{res['avg_ms']:<10.2f} | {res['median_ms']:<11.2f} | "
            f"{res['successes']:<7} | {res['failures']:<6} | {res['timeouts']:<7} |"
        )
        
    print("\n--- Test Complete ---\n")


# ============================================================
#                      MAIN LOGIC
# ============================================================

async def main():
    """
    Main asynchronous function to orchestrate all tests.
    """
    
    # 1. Format the prompt
    prompt = PROMPT_TEMPLATE.format(clue=CLUE, words=str(WORDS))

    # 2. Create a list of all tasks to run
    tasks = []

    # Add Gemini models to the task list
    for model_name in GEMINI_MODELS:
        client = GeminiClient(model_name=model_name)
        tasks.append(
            run_model_test(client, "Gemini", prompt, NUM_REQUESTS, REQUEST_TIMEOUT)
        )

    # Add SiliconFlow models to the task list
    for model_name in SILICONFLOW_MODELS:
        # We must create a new client *instance* for each model
        client = OpenAIClient(model_name=model_name)
        tasks.append(
            run_model_test(client, "SiliconFlow", prompt, NUM_REQUESTS, REQUEST_TIMEOUT)
        )
        
    # You could add other providers here (e.g., Anthropic) in the same way

    print(f"--- Starting All-Model Latency Test ---")
    print(f"Total Models to Test: {len(tasks)}")
    print(f"Requests per Model:   {NUM_REQUESTS}")
    print(f"Request Timeout:      {REQUEST_TIMEOUT}s")
    print("This may take several minutes...\n")

    # 3. Run all tasks concurrently
    # asyncio.gather runs all tasks in the list at the same time
    # and collects all their return values (our stat dictionaries)
    start_time = time.perf_counter()
    
    # We set return_exceptions=True so if one test crashes hard,
    # it doesn't stop all the other tests.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.perf_counter()
    
    print(f"\n...All tests completed in {(end_time - start_time):.2f} seconds.")

    # 4. Filter out any exceptions and print the final report
    final_results = [res for res in results if isinstance(res, dict)]
    exceptions = [res for res in results if not isinstance(res, dict)]

    print_summary_report(final_results)
    
    if exceptions:
        print(f"\n--- ⚠️ CRITICAL ERRORS ---")
        print(f"{len(exceptions)} tasks failed to even start (e.g., auth errors).")
        for exc in exceptions:
            print(f"- {exc}")


if __name__ == "__main__":
    # This is the standard way to run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user. Exiting.")
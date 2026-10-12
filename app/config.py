import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Backend service URLs
BACKEND_URLS = {
    "model_a": "http://127.0.0.1:8001",
    "model_b": "http://127.0.0.1:8002",
    "model_c": "http://127.0.0.1:8003",
    "ollama_qwen3": "http://127.0.0.1:8004",
}

# Time-scale factor for faster test runs and live demos (preserves relative ratios)
TIME_SCALE = float(os.getenv("TIME_SCALE", "1.0"))

# Queue and concurrency configuration
QUEUE_MAX_SIZE = 50
# Maximum concurrent requests per backend worker (controls queue formation)
BACKEND_CONCURRENCY = int(os.getenv("BACKEND_CONCURRENCY", "1"))

# Scheduling policies
POLICY_FIFO = "fifo"
POLICY_TOKEN_AWARE = "token_aware"
POLICY_ADAPTIVE = "adaptive"
POLICY_ADAPTIVE_LEAST_CONN = "adaptive_least_conn"
POLICY_PRIORITY_ONLY = "priority_only"
POLICY_SJF = "sjf"
POLICY_ADAPTIVE_ORACLE = "adaptive_oracle"
POLICY_ADAPTIVE_PRED = "adaptive_pred"
POLICY_ADAPTIVE_NO_AGING = "adaptive_no_aging"
POLICY_ADAPTIVE_NO_COST = "adaptive_no_cost"

# Profile priority scores
PROFILE_SCORES = {
    "interactive": 100,
    "normal": 50,
    "batch": 10,
}

# Base token delay per backend (ms per token)
TOKEN_DELAY_MS = {
    "model_a": 50,
    "model_b": 100,
    "model_c": 200,
}

# Adaptive policy weights (normalized scoring formulation)
# Score = (ProfileScore * W_profile) + (WaitSeconds * W_wait) - (EstCostSec * W_cost) + StarvationBonus
ADAPTIVE_PROFILE_WEIGHT = 1.0
ADAPTIVE_WAIT_WEIGHT = 5.0
ADAPTIVE_COST_WEIGHT = 2.0
STARVATION_WAIT_THRESHOLD = 30.0 * TIME_SCALE
STARVATION_BONUS = 1000.0

# Circuit Breaker thresholds
CB_FAILURE_THRESHOLD = 3       # Consecutive failures before opening circuit
CB_OPEN_DURATION = 5.0 * TIME_SCALE  # Seconds circuit remains open before half-open probe
CB_HALF_OPEN_PROBES = 1        # Successful probes needed to close circuit

# Benchmark configuration
BENCHMARK_WORKLOAD_SIZE = 25
BENCHMARK_SEED = 42

# Results directory
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WORKLOADS_DIR = RESULTS_DIR / "workloads"
WORKLOADS_DIR.mkdir(exist_ok=True)
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# Backend Simulation Model ("simple" or "batched")
BACKEND_MODEL = os.getenv("BACKEND_MODEL", "simple")
BATCHED_ALPHA = float(os.getenv("BATCHED_ALPHA", "0.2"))   # Per-active-sequence delay scaling
BATCHED_BETA = float(os.getenv("BATCHED_BETA", "0.0005"))  # Prefill delay (seconds per prompt token)

# SLO Targets (in seconds, scaled with TIME_SCALE)
SLO_TARGETS = {
    "interactive": {"ttft": 2.0 * TIME_SCALE, "completion": 6.0 * TIME_SCALE},
    "normal": {"ttft": 5.0 * TIME_SCALE, "completion": 18.0 * TIME_SCALE},
    "batch": {"ttft": 15.0 * TIME_SCALE, "completion": 50.0 * TIME_SCALE},
}

# Admission Control Configuration (Off by default)
ENABLE_ADMISSION_CONTROL = os.getenv("ENABLE_ADMISSION_CONTROL", "false").lower() == "true"
ADMISSION_CONTROL_FACTOR = float(os.getenv("ADMISSION_CONTROL_FACTOR", "1.5"))

# Ollama LLM Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://127.0.0.1:11434/api/tags")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-q4_K_M")


def resolve_ollama_model(preferred: str | None = None) -> str:
    """Resolve actual installed Ollama model tag dynamically."""
    target = preferred or OLLAMA_MODEL
    try:
        import httpx
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(OLLAMA_TAGS_URL)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                if target in models:
                    return target
                # Check for prefix or substring match
                for m in models:
                    if m.startswith(target) or target in m:
                        return m
                # If target has "qwen", match any qwen model
                if "qwen" in target.lower():
                    for m in models:
                        if "qwen" in m.lower():
                            return m
                if models:
                    return models[0]
    except Exception:
        pass
    return target


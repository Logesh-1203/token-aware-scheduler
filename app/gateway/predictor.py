"""Online EWMA Output-Length Predictor and Evaluation Harness.

Provides online estimation of actual output token lengths based on
exponentially weighted moving averages (EWMA) per (profile, prompt_bucket).
Includes statistical evaluation functions (MAE, MAPE, Spearman correlation)
and noise injection for robustness testing.
"""

from typing import Dict, Tuple, List, Any, Optional
import random
import numpy as np


class OnlineTokenPredictor:
    """Lightweight online predictor for LLM output token counts."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        # Key: (profile, prompt_bucket) -> EWMA float
        self._history: Dict[Tuple[str, str], float] = {}
        # Initial priors by profile if no observations exist
        self._default_priors = {
            "interactive": 15.0,
            "normal": 45.0,
            "batch": 120.0,
        }

    def _get_prompt_bucket(self, prompt: str) -> str:
        """Classify prompt into length buckets: short, medium, long."""
        length = len(prompt.strip()) if prompt else 0
        if length < 50:
            return "short"
        elif length < 150:
            return "medium"
        return "long"

    def observe(self, profile: str, prompt: str, actual_tokens: int) -> None:
        """Update EWMA estimate with actual observed output tokens."""
        if actual_tokens <= 0:
            return
        bucket = self._get_prompt_bucket(prompt)
        key = (profile, bucket)
        if key in self._history:
            self._history[key] = (1.0 - self.alpha) * self._history[key] + self.alpha * float(actual_tokens)
        else:
            self._history[key] = float(actual_tokens)

    def predict(self, profile: str, prompt: str, max_output_tokens: int) -> int:
        """Predict actual output tokens, clamped strictly to [1, max_output_tokens]."""
        if max_output_tokens <= 1:
            return 1
        bucket = self._get_prompt_bucket(prompt)
        key = (profile, bucket)
        if key in self._history:
            est = self._history[key]
        else:
            est = self._default_priors.get(profile, 30.0)

        pred = int(round(est))
        # Hard bounds: [1, max_output_tokens]
        return max(1, min(pred, max_output_tokens))

    def reset(self) -> None:
        """Clear all historical observations."""
        self._history.clear()

    def get_state(self) -> Dict[str, float]:
        """Return human-readable table of current EWMA estimates."""
        return {f"{k[0]}:{k[1]}": round(v, 2) for k, v in self._history.items()}


# Global predictor instance
default_predictor = OnlineTokenPredictor()


def calculate_mae(predictions: List[float], actuals: List[float]) -> float:
    """Mean Absolute Error."""
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0
    arr_p = np.array(predictions, dtype=float)
    arr_a = np.array(actuals, dtype=float)
    return round(float(np.mean(np.abs(arr_p - arr_a))), 4)


def calculate_mape(predictions: List[float], actuals: List[float]) -> float:
    """Mean Absolute Percentage Error (clamped to avoid division by zero)."""
    if not predictions or not actuals or len(predictions) != len(actuals):
        return 0.0
    arr_p = np.array(predictions, dtype=float)
    arr_a = np.array(actuals, dtype=float)
    denom = np.maximum(arr_a, 1.0)
    return round(float(np.mean(np.abs(arr_p - arr_a) / denom)) * 100.0, 2)


def calculate_spearman(predictions: List[float], actuals: List[float]) -> float:
    """Spearman rank correlation coefficient computed via NumPy."""
    if len(predictions) < 2 or len(actuals) < 2 or len(predictions) != len(actuals):
        return 0.0
    arr_p = np.array(predictions, dtype=float)
    arr_a = np.array(actuals, dtype=float)

    # Convert values to ranks
    rank_p = np.argsort(np.argsort(arr_p)).astype(float)
    rank_a = np.argsort(np.argsort(arr_a)).astype(float)

    std_p = np.std(rank_p)
    std_a = np.std(rank_a)
    if std_p < 1e-9 or std_a < 1e-9:
        return 0.0

    cov = np.mean((rank_p - np.mean(rank_p)) * (rank_a - np.mean(rank_a)))
    corr = cov / (std_p * std_a)
    return round(float(corr), 4)


def evaluate_predictions(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """Calculate MAE, MAPE, and Spearman rank correlation."""
    return {
        "mae": calculate_mae(predictions, actuals),
        "mape_pct": calculate_mape(predictions, actuals),
        "spearman_rank_corr": calculate_spearman(predictions, actuals),
        "count": len(predictions),
    }


def inject_prediction_noise(
    predictions: List[int],
    noise_level: float,
    max_tokens_list: List[int],
    seed: int = 42,
) -> List[int]:
    """Inject noise into predictions: pred_noisy = (1 - eta)*pred + eta*random_token.
    noise_level is eta in [0.0, 1.0].
    """
    if noise_level <= 0.0:
        return list(predictions)
    rng = random.Random(seed)
    noisy = []
    for pred, max_tok in zip(predictions, max_tokens_list):
        if max_tok <= 1:
            noisy.append(1)
            continue
        rand_val = rng.randint(1, max_tok)
        val = (1.0 - noise_level) * float(pred) + noise_level * float(rand_val)
        clamped = max(1, min(int(round(val)), max_tok))
        noisy.append(clamped)
    return noisy

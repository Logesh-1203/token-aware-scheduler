import pytest
import asyncio
import tempfile
from pathlib import Path
import numpy as np

from app.gateway.predictor import (
    OnlineTokenPredictor,
    calculate_mae,
    calculate_mape,
    calculate_spearman,
    evaluate_predictions,
    inject_prediction_noise,
)
from app.gateway.benchmark import (
    generate_deterministic_workload,
    save_workload,
    load_workload,
    calculate_summary_metrics,
    bootstrap_confidence_interval,
)
from app.gateway.scheduler import (
    compute_priority_only_score,
    compute_sjf_score,
    compute_adaptive_score,
    check_admission_control,
    reset_scheduler_state,
    _queue,
)
import app.config as cfg


def test_workload_reproducibility_and_serialization():
    """Verify deterministic workload generation and JSON save/load round-trip."""
    w1 = generate_deterministic_workload(size=10, seed=123, arrival_process="poisson", workload_preset="balanced")
    w2 = generate_deterministic_workload(size=10, seed=123, arrival_process="poisson", workload_preset="balanced")

    assert len(w1) == 10
    assert w1 == w2, "Workloads generated with same seed must be identical"

    # Verify presence of true tokens
    for item in w1:
        assert "sim_true_output_tokens" in item
        assert 1 <= item["sim_true_output_tokens"] <= item["max_output_tokens"]
        assert "arrival_offset" in item

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_workload.json"
        saved = save_workload(w1, tmp_path)
        loaded = load_workload(saved)
        assert loaded == w1, "Loaded workload JSON must match original"


def test_online_predictor_bounds_and_updates():
    """Verify predictor respects [1, max_output_tokens] bounds and EWMA updates."""
    pred = OnlineTokenPredictor(alpha=0.5)

    # Initial prediction without observations should be bounded
    val = pred.predict("interactive", "short prompt", max_output_tokens=20)
    assert 1 <= val <= 20

    val_clamped = pred.predict("interactive", "short prompt", max_output_tokens=5)
    assert val_clamped <= 5

    # Observe true output tokens
    pred.observe("interactive", "short prompt", actual_tokens=10)
    pred.observe("interactive", "short prompt", actual_tokens=10)
    val2 = pred.predict("interactive", "short prompt", max_output_tokens=50)
    assert 9 <= val2 <= 11

    # Reset clears history
    pred.reset()
    assert len(pred._history) == 0


def test_prediction_metrics_mae_mape_spearman():
    """Verify MAE, MAPE, and Spearman correlation calculations."""
    preds = [10.0, 20.0, 30.0, 40.0]
    actuals = [12.0, 18.0, 32.0, 38.0]

    eval_res = evaluate_predictions(preds, actuals)
    assert eval_res["mae"] == 2.0
    assert eval_res["mape_pct"] > 0.0
    assert eval_res["spearman_rank_corr"] > 0.95
    assert eval_res["count"] == 4


def test_prediction_noise_injection_degradation():
    """Verify noise injection increases MAE monotonically as noise increases."""
    actuals = [15, 45, 120, 30, 60]
    preds = [15, 45, 120, 30, 60]  # Perfect initial predictions
    max_tokens = [30, 80, 200, 50, 100]

    noise_0 = inject_prediction_noise(preds, 0.0, max_tokens, seed=42)
    noise_25 = inject_prediction_noise(preds, 0.25, max_tokens, seed=42)
    noise_50 = inject_prediction_noise(preds, 0.50, max_tokens, seed=42)
    noise_100 = inject_prediction_noise(preds, 1.00, max_tokens, seed=42)

    mae_0 = calculate_mae(noise_0, actuals)
    mae_25 = calculate_mae(noise_25, actuals)
    mae_50 = calculate_mae(noise_50, actuals)
    mae_100 = calculate_mae(noise_100, actuals)

    assert mae_0 == 0.0
    assert mae_25 <= mae_50 <= mae_100


def test_slo_attainment_and_goodput_calculation():
    """Verify calculate_summary_metrics computes SLO attainment and goodput."""
    mock_results = [
        {
            "final_status": "completed",
            "completion_latency": 2.0,
            "ttft": 0.5,
            "queue_wait": 0.2,
            "profile": "interactive",
            "slo_met": True,
        },
        {
            "final_status": "completed",
            "completion_latency": 4.0,
            "ttft": 1.0,
            "queue_wait": 0.5,
            "profile": "interactive",
            "slo_met": False,
        },
    ]

    summary = calculate_summary_metrics(mock_results, "adaptive", total_duration=5.0)
    assert summary["slo_attainment_pct"] == 50.0
    assert summary["goodput_req_per_sec"] == 0.20  # 1 meeting SLO / 5.0 s
    assert summary["completed_requests"] == 2


def test_bootstrap_confidence_interval():
    """Verify 95% bootstrap confidence interval computation."""
    data = [10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 10.1]
    mean, ci_low, ci_high = bootstrap_confidence_interval(data, n_resamples=1000, ci=0.95, seed=42)
    assert ci_low <= mean <= ci_high
    assert 9.0 <= ci_low
    assert ci_high <= 12.0


def test_scheduler_policy_ablations_and_scores():
    """Verify scoring logic for priority_only, sjf, and adaptive ablations."""
    now = 100.0
    req = {
        "enqueued_at": now - 35.0,  # 35s in queue (above starvation threshold of 30s)
        "profile": "batch",
        "max_output_tokens": 150,
        "predicted_tokens": 80,
        "sim_true_output_tokens": 70,
    }

    # 1. Priority Only: only profile score, ignores wait and tokens
    p_score, p_breakdown = compute_priority_only_score(req, now)
    assert p_score == cfg.PROFILE_SCORES["batch"]

    # 2. SJF: score is negative of tokens
    sjf_score, sjf_breakdown = compute_sjf_score(req, now)
    assert sjf_score == -150.0

    # 3. Adaptive with aging: starvation bonus triggered
    adapt_score, adapt_bd = compute_adaptive_score(req, now, policy_variant=cfg.POLICY_ADAPTIVE)
    assert adapt_bd["starvation_bonus"] == cfg.STARVATION_BONUS

    # 4. Adaptive No Aging ablation: aging score is 0 and starvation bonus is 0!
    no_aging_score, no_aging_bd = compute_adaptive_score(req, now, policy_variant=cfg.POLICY_ADAPTIVE_NO_AGING)
    assert no_aging_bd["aging_score"] == 0.0
    assert no_aging_bd["starvation_bonus"] == 0.0
    assert no_aging_score < adapt_score

    # 5. Adaptive No Cost ablation: cost penalty is 0!
    no_cost_score, no_cost_bd = compute_adaptive_score(req, now, policy_variant=cfg.POLICY_ADAPTIVE_NO_COST)
    assert no_cost_bd["cost_penalty"] == 0.0

    # 6. Adaptive Pred variant: uses predicted_tokens (80) rather than max_output_tokens (150)
    pred_score, pred_bd = compute_adaptive_score(req, now, policy_variant=cfg.POLICY_ADAPTIVE_PRED)
    assert pred_bd["effective_tokens"] == 80
    assert pred_bd["cost_penalty"] < adapt_bd["cost_penalty"]


def test_admission_control_shedding():
    """Verify admission control sheds low-priority requests when queue is overloaded."""
    reset_scheduler_state()
    # Default is False
    cfg.ENABLE_ADMISSION_CONTROL = False
    assert check_admission_control("batch") is True

    # Enable admission control
    cfg.ENABLE_ADMISSION_CONTROL = True
    try:
        # Fill mock queue with 16 requests
        _queue.clear()
        for i in range(16):
            _queue.append({"request_id": f"q_{i}", "profile": "batch"})

        # Batch request should be shed
        assert check_admission_control("batch") is False
        # Interactive request should still be admitted
        assert check_admission_control("interactive") is True
    finally:
        cfg.ENABLE_ADMISSION_CONTROL = False
        reset_scheduler_state()

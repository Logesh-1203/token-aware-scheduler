import pytest
import asyncio
import io
import csv
import tempfile
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from app.gateway.main import app as gateway_app
from app.gateway.scheduler import (
    estimate_prompt_tokens,
    choose_backend_for_request,
    reset_scheduler_state,
    health_manager,
    BACKEND_URLS,
    POLICY_ADAPTIVE,
    POLICY_FIFO,
    _active_requests,
)
from app.gateway.traces import DecisionTraceManager, trace_manager
from app.gateway.benchmark import (
    run_side_by_side_benchmark,
    calculate_summary_metrics,
)
from app.metrics_store import MetricsStore


@pytest.fixture(autouse=True)
def clean_state():
    reset_scheduler_state()
    yield
    reset_scheduler_state()


# ---------------------------------------------------------------------------
# 1. Prompt Token Demand Estimation
# ---------------------------------------------------------------------------
def test_estimate_prompt_tokens_heuristics():
    # Empty / whitespace fallback
    res_empty = estimate_prompt_tokens("")
    assert res_empty["estimated_tokens"] == 1
    assert res_empty["method"] == "fallback_empty"
    assert res_empty["is_exact"] is False
    assert res_empty["char_count"] == 0
    assert res_empty["word_count"] == 0

    res_spaces = estimate_prompt_tokens("   \n\t  ")
    assert res_spaces["estimated_tokens"] == 1
    assert res_spaces["method"] == "fallback_empty"

    # Multi-word prompt
    prompt = "Explain token-aware scheduling in simple Tamil"
    res = estimate_prompt_tokens(prompt)
    assert res["method"] == "word_char_heuristic_1.33w_4c"
    assert res["word_count"] == 6
    assert res["char_count"] == len(prompt)
    assert res["estimated_tokens"] >= 6
    assert res["is_exact"] is False


# ---------------------------------------------------------------------------
# 2. Adaptive Candidate Scoring & Cost Estimation
# ---------------------------------------------------------------------------
def test_adaptive_candidate_scoring_and_selection():
    # Ensure all backends are healthy
    for b in BACKEND_URLS:
        health_manager.record_success(b, latency_sec=0.1, tokens=10)

    # Set mock ms_per_token: model_a faster (50ms) than model_b (100ms) and model_c (200ms)
    health_manager._observed_ms_per_token["model_a"] = 50.0
    health_manager._observed_ms_per_token["model_b"] = 100.0
    health_manager._observed_ms_per_token["model_c"] = 200.0

    best_b, estimates, rejected = choose_backend_for_request(
        policy=POLICY_ADAPTIVE,
        max_output_tokens=50,
        candidate_backends=list(BACKEND_URLS.keys()),
        prompt="Test prompt",
        profile="interactive",
    )

    # Model A should be selected because it has the lowest service time and zero queue depth
    assert best_b == "model_a"
    assert "model_a" in estimates
    assert "model_b" in estimates
    assert "model_c" in estimates

    # Check structure of candidate estimates
    est_a = estimates["model_a"]
    assert "active" in est_a
    assert "ms_per_token" in est_a
    assert "observed_tps" in est_a
    assert "slot_wait_sec" in est_a
    assert "token_work_sec" in est_a
    assert "estimated_completion_sec" in est_a
    assert "routing_score" in est_a

    # Model A's estimated completion time should be strictly less than Model C's
    assert estimates["model_a"]["estimated_completion_sec"] < estimates["model_c"]["estimated_completion_sec"]
    assert len(rejected) == 0


def test_adaptive_circuit_breaker_rejection():
    # Force trip circuit breaker on model_a
    for _ in range(5):
        health_manager.record_failure("model_a", reason="500 Internal Error")

    assert not health_manager.is_available("model_a")

    best_b, estimates, rejected = choose_backend_for_request(
        policy=POLICY_ADAPTIVE,
        max_output_tokens=30,
        candidate_backends=list(BACKEND_URLS.keys()),
    )

    # Model A should be rejected with circuit_open status and not selected
    assert "model_a" in rejected
    assert "circuit_open" in rejected["model_a"]
    assert best_b != "model_a"


# ---------------------------------------------------------------------------
# 3. Dynamic Decision Explanations
# ---------------------------------------------------------------------------
def test_dynamic_explanation_generation():
    tm = DecisionTraceManager(capacity=10)

    # Successful adaptive dispatch explanation
    exp = tm.generate_explanation(
        policy="adaptive",
        profile="interactive",
        selected_backend="model_a",
        queue_wait=0.05,
        score_breakdown={"total_score": 95.0},
        candidate_estimates={
            "model_a": {"estimated_completion_sec": 0.25, "ms_per_token": 50.0, "active": 0, "observed_tps": 20.0},
            "model_b": {"estimated_completion_sec": 0.50},
        },
        rejected_reasons={"model_c": "circuit_open"},
        success=True,
        prompt_tokens=15,
        max_output_tokens=40,
        fallback_occurred=False,
    )

    assert "Adaptive policy scheduled interactive request" in exp
    assert "model_a" in exp
    assert "0.25s" in exp
    assert "est input: 15 tok" in exp
    assert "max output: 40 tok" in exp
    assert "model_b (0.50s)" in exp
    assert "model_c (circuit_open)" in exp
    assert "Fallback" not in exp

    # Fallback explanation
    exp_fallback = tm.generate_explanation(
        policy="adaptive",
        profile="interactive",
        selected_backend="model_b",
        queue_wait=0.10,
        score_breakdown={"total_score": 80.0},
        candidate_estimates={"model_b": {"estimated_completion_sec": 0.40, "ms_per_token": 80.0, "active": 0}},
        rejected_reasons={},
        success=True,
        fallback_occurred=True,
    )
    assert "[Fallback: primary backend failed; successfully rerouted to secondary backend]" in exp_fallback


# ---------------------------------------------------------------------------
# 4. Side-by-Side Benchmark Engine & Warmup Exclusion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_side_by_side_benchmark_simulated():
    # Mock streaming function for benchmark simulation
    class MockStreamClient:
        async def aclose(self):
            pass

    class MockStreamResponse:
        def __init__(self, tokens=5):
            self.tokens = tokens

        async def aiter_text(self):
            for i in range(self.tokens):
                yield f'{{"type":"token","text":"tok{i} "}}\n'
            yield '{"type":"done","tokens_generated":5}\n'

    async def mock_stream_fn(url, payload):
        return MockStreamClient(), MockStreamResponse(tokens=5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_bench.db"
        test_store = MetricsStore(path=str(db_path))
        try:
            result = await run_side_by_side_benchmark(
                baseline_policy=POLICY_FIFO,
                adaptive_policy=POLICY_ADAPTIVE,
                workload_size=4,
                workload_preset="quick",
                mode="SIMULATED",
                seed=42,
                warmup_count=2,
                concurrency=1,
                stream_from_fn=mock_stream_fn,
                metrics_store=test_store,
            )

            assert "run_id" in result
            assert result["baseline_policy"] == POLICY_FIFO
            assert result["adaptive_policy"] == POLICY_ADAPTIVE
            assert result["mode"] == "SIMULATED"
            assert result["warmup_count"] == 2
            assert result["workload_size"] == 4

            # Verification: raw results contain 4 records per policy = 8 evaluated records (warmup excluded!)
            assert len(result["raw_results"]) == 8
            baseline_recs = [r for r in result["raw_results"] if r["benchmark_policy_role"] == "baseline"]
            adaptive_recs = [r for r in result["raw_results"] if r["benchmark_policy_role"] == "adaptive"]
            assert len(baseline_recs) == 4
            assert len(adaptive_recs) == 4

            # Verify comparison metric schema
            comp = result["comparison"]
            assert "mean_latency_sec" in comp
            assert "throughput_req_per_sec" in comp
            assert "tokens_per_second_aggregate" in comp
            assert comp["mean_latency_sec"]["favorable_direction"] == "lower"
            assert comp["throughput_req_per_sec"]["favorable_direction"] == "higher"
            assert comp["mean_latency_sec"]["winner"] in ("baseline", "adaptive", "tie")

            # Verify persisted run in test_store
            stored_run = test_store.get_benchmark_run(result["run_id"])
            assert stored_run is not None
            assert stored_run["run_id"] == result["run_id"]
            assert stored_run["baseline_policy"] == POLICY_FIFO
            assert stored_run["adaptive_policy"] == POLICY_ADAPTIVE
        finally:
            test_store.close()


# ---------------------------------------------------------------------------
# 5. Benchmark Persistence & CSV Export
# ---------------------------------------------------------------------------
def test_metrics_store_benchmark_crud_and_csv():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MetricsStore(path=str(Path(tmp_dir) / "bench.db"))
        try:
            run_data = {
                "run_id": "test_run_123",
                "baseline_policy": "fifo",
                "adaptive_policy": "adaptive",
                "mode": "SIMULATED",
                "workload_preset": "balanced",
                "request_count": 2,
                "concurrency": 1,
                "warmup_count": 1,
                "summary": {
                    "baseline": {"mean_latency_sec": 0.50, "throughput_req_per_sec": 2.0},
                    "adaptive": {"mean_latency_sec": 0.35, "throughput_req_per_sec": 2.8},
                    "comparison": {"mean_latency_sec": {"winner": "adaptive", "pct_change": -30.0}},
                },
                "raw_results": [
                    {"benchmark_policy_role": "baseline", "request_id": "req-1", "completion_latency": 0.50},
                    {"benchmark_policy_role": "adaptive", "request_id": "req-1", "completion_latency": 0.35},
                ],
            }

            # Save run
            store.save_benchmark_run(run_data)

            # List runs
            runs = store.get_benchmark_runs()
            assert len(runs) >= 1
            assert runs[0]["run_id"] == "test_run_123"

            # Detail
            detail = store.get_benchmark_run("test_run_123")
            assert detail is not None
            assert detail["mode"] == "SIMULATED"
            assert len(detail["raw_results"]) == 2
            assert detail["summary"]["baseline"]["mean_latency_sec"] == 0.50

            # Export CSV
            csv_str = store.export_benchmark_csv("test_run_123")
            assert csv_str is not None
            reader = list(csv.DictReader(io.StringIO(csv_str)))
            assert len(reader) == 2
            assert reader[0]["request_id"] == "req-1"
            assert reader[0]["benchmark_policy_role"] == "baseline"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 6. API Endpoints: /scheduler/status and /benchmark/*
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scheduler_status_and_benchmark_endpoints(monkeypatch):
    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GET /scheduler/status
        status_resp = await client.get("/scheduler/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert "metrics" in status_data
        assert "backends" in status_data
        assert "details" in status_data
        assert "recent_traces" in status_data

        # 2. GET /benchmark/history
        hist_resp = await client.get("/benchmark/history")
        assert hist_resp.status_code == 200
        assert "runs" in hist_resp.json()

        # 3. POST /benchmark/compare with mocked stream to avoid external network
        class MockClient:
            async def aclose(self):
                pass

        class MockResp:
            async def aiter_text(self):
                yield '{"type":"token","text":"hello "}\n'
                yield '{"type":"done","tokens_generated":1}\n'

        async def fake_stream(url, payload):
            return MockClient(), MockResp()

        monkeypatch.setattr("app.gateway.router._stream_from_backend", fake_stream)

        compare_resp = await client.post(
            "/benchmark/compare",
            json={
                "baseline_policy": "fifo",
                "adaptive_policy": "adaptive",
                "workload_size": 2,
                "workload_preset": "quick",
                "mode": "SIMULATED",
                "seed": 99,
                "warmup_count": 0,
            },
        )
        assert compare_resp.status_code == 200
        comp_data = compare_resp.json()
        assert "run_id" in comp_data
        assert comp_data["mode"] == "SIMULATED"
        assert "comparison" in comp_data
        run_id = comp_data["run_id"]

        # 4. GET /benchmark/history/{run_id}
        detail_resp = await client.get(f"/benchmark/history/{run_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["run_id"] == run_id

        # 5. GET /benchmark/history/{run_id}/export.csv
        csv_resp = await client.get(f"/benchmark/history/{run_id}/export.csv")
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.headers["content-type"]
        assert len(csv_resp.text) > 0

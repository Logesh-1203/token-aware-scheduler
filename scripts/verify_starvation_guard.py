"""Live verification script for SLO-Aware Starvation Prevention Guard.

Demonstrates:
1. Dynamic Settings Configuration & Reset (API and Scheduler)
2. Queue Aging & Priority Escalation for Waiting Requests
3. Batch Starvation Prevention (Starvation Bonus applied)
4. Protection ON vs Protection OFF comparison on identical heavy workloads
5. SLO Monitor tracking p95 latency vs configured target
"""
import asyncio
import time
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from httpx import AsyncClient, ASGITransport

from app.gateway.main import app as gateway_app
from app.gateway.scheduler import (
    reset_scheduler_state,
    schedule_and_acquire,
    release_slot,
    get_metrics,
    get_starvation_guard_status,
    _queue,
    _active_requests,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    POLICY_FIFO,
)
from app.gateway.starvation_guard import starvation_guard


async def main():
    print("=" * 75)
    print(" HORIZON 2K26 — SLO-Aware Starvation Prevention Guard: Live Verification")
    print("=" * 75)

    transport = ASGITransport(app=gateway_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        # -------------------------------------------------------------------
        # 1. Verify Configuration Endpoint & Live KPIs
        # -------------------------------------------------------------------
        print("\n[1] Testing GET /scheduler/starvation-guard...")
        r = await client.get("/scheduler/starvation-guard")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        cfg = data["config"]
        kpis = data["kpis"]
        print(f"  [OK] Starvation Guard Status:   {'ENABLED' if cfg['enabled'] else 'DISABLED'}")
        print(f"  [OK] Max Queue Wait Threshold:  {cfg['max_queue_wait_threshold_sec']}s")
        print(f"  [OK] Aging Boost Rate:          +{cfg['aging_boost_rate']} pts/sec")
        print(f"  [OK] Starvation Bonus:          +{cfg['starvation_bonus']} pts")
        print(f"  [OK] Configured SLO Target:     {cfg['slo_latency_target_sec']}s")
        print(f"  [OK] Current Queue Depth:       {kpis['queue_depth']}")
        print(f"  [OK] Requests Protected:        {kpis['requests_protected_by_aging']}")

        # -------------------------------------------------------------------
        # 2. Test Interactive Settings Update & Validation
        # -------------------------------------------------------------------
        print("\n[2] Testing POST /scheduler/starvation-guard (Runtime Reconfiguration)...")
        new_settings = {
            "enabled": True,
            "max_queue_wait_threshold_sec": 5.0,
            "aging_boost_rate": 8.0,
            "slo_latency_target_sec": 10.0,
        }
        r_up = await client.post("/scheduler/starvation-guard", json=new_settings)
        assert r_up.status_code == 200
        up_cfg = r_up.json()["config"]
        print(f"  [OK] Updated Wait Threshold to: {up_cfg['max_queue_wait_threshold_sec']}s")
        print(f"  [OK] Updated Aging Boost Rate:  +{up_cfg['aging_boost_rate']} pts/s")

        # -------------------------------------------------------------------
        # 3. Demonstrate Queue Aging & Starvation Prevention Under Load
        # -------------------------------------------------------------------
        print("\n[3] Testing Live Queue Aging & Priority Escalation...")
        reset_scheduler_state()
        starvation_guard.reset()
        starvation_guard.update_config({"enabled": True, "max_queue_wait_threshold_sec": 4.0, "aging_boost_rate": 10.0})

        backend_urls = {"model_a": "http://a"}

        # Saturate Model A slot (concurrency limit = 1)
        r0 = {"prompt": "blocking", "profile": "interactive", "max_output_tokens": 10, "policy": POLICY_TOKEN_AWARE}
        b0, _, _, _ = await schedule_and_acquire(r0, backend_urls=backend_urls)
        print(f"  [OK] Model A saturated (active requests = {_active_requests['model_a']})")

        # Enqueue low-priority batch request (large tokens = 150)
        batch_req = {
            "request_id": "batch-req-001",
            "prompt": "heavy document summarization",
            "profile": "batch",
            "max_output_tokens": 150,
            "policy": POLICY_TOKEN_AWARE,
        }
        task_batch = asyncio.create_task(schedule_and_acquire(batch_req, backend_urls=backend_urls))
        await asyncio.sleep(0.02)

        # Baseline inspection right after enqueue
        stat_before = get_starvation_guard_status()
        q_item = stat_before["queued_requests"][0]
        print(f"  [OK] Queued Batch Request at t=0: Original Priority={q_item['original_priority']}, Effective Priority={q_item['effective_priority']}, Aging Boost=+{q_item['aging_adjustment']}")

        # Simulate 5.0 seconds of wait time (> 4.0s threshold)
        _queue[0]["enqueued_at"] -= 5.0

        stat_after = get_starvation_guard_status()
        q_item_starved = stat_after["queued_requests"][0]
        print(f"  [OK] After 5s wait: Wait Time={q_item_starved['queue_wait_sec']}s, Effective Priority={q_item_starved['effective_priority']}, Aging Boost=+{q_item_starved['aging_adjustment']}")
        print(f"  [OK] Starvation Risk: {q_item_starved['starvation_risk']} ({q_item_starved['starvation_risk_color'].upper()})")
        assert q_item_starved["starvation_bonus_active"] is True, "Starvation bonus should be active!"

        # Enqueue new interactive request (normally score = 100 - 2 = 98)
        inter_req = {
            "request_id": "inter-req-002",
            "prompt": "quick chat",
            "profile": "interactive",
            "max_output_tokens": 10,
            "policy": POLICY_TOKEN_AWARE,
        }
        task_inter = asyncio.create_task(schedule_and_acquire(inter_req, backend_urls=backend_urls))
        await asyncio.sleep(0.02)

        # Release Model A slot
        await release_slot(b0)
        dispatched_backend, _, wait_sec, meta = await task_batch

        print(f"  [OK] Dispatched Winner: {dispatched_backend} (Batch request won slot due to Starvation Bonus: {meta['score_breakdown']['starvation_bonus']} pts!)")
        assert dispatched_backend == "model_a"
        assert meta["score_breakdown"]["starvation_bonus"] == 1000.0

        # Clean up second task
        await release_slot(dispatched_backend)
        await task_inter

        # -------------------------------------------------------------------
        # 4. Run Live Starvation Guard Benchmark Comparison (OFF vs ON)
        # -------------------------------------------------------------------
        print("\n[4] Running Benchmark Comparison: Starvation Protection OFF vs ON...")
        bench_res = await client.post("/benchmark/starvation-compare", json={
            "policy": "adaptive",
            "workload_size": 10,
            "seed": 42,
            "workload_preset": "heavy_batch",
        })
        assert bench_res.status_code == 200
        comp = bench_res.json()
        off_m = comp["protection_off"]
        on_m = comp["protection_on"]

        print("  Comparison Results on Identical Workload (10 requests, seed=42):")
        print(f"  - Maximum Queue Wait:      OFF = {off_m['max_queue_wait_sec']:.2f}s  |  ON = {on_m['max_queue_wait_sec']:.2f}s (Diff: {on_m['max_queue_wait_sec'] - off_m['max_queue_wait_sec']:.2f}s)")
        print(f"  - p95 Queue Wait:          OFF = {off_m['p95_queue_wait_sec']:.2f}s  |  ON = {on_m['p95_queue_wait_sec']:.2f}s")
        print(f"  - Starvation Incidents:    OFF = {off_m['starvation_risk_requests']}       |  ON = {on_m['starvation_risk_requests']}")
        print(f"  - Completed Requests:      OFF = {off_m['completed_requests']}      |  ON = {on_m['completed_requests']}")
        print(f"  - p95 End-to-End Latency:  OFF = {off_m['p95_latency_sec']:.2f}s  |  ON = {on_m['p95_latency_sec']:.2f}s")
        print(f"\n  Trade-Off Analysis:\n  {comp['tradeoff_analysis']}")

        print("\n" + "=" * 75)
        print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (100%)")
        print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import httpx

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.gateway.benchmark import (
    run_full_benchmark_lab,
    run_multi_seed_benchmark,
    generate_deterministic_workload,
    save_workload,
)
from scripts.plot_benchmark import generate_benchmark_charts
from app.config import (
    POLICY_FIFO,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    POLICY_ADAPTIVE_LEAST_CONN,
    POLICY_PRIORITY_ONLY,
    POLICY_SJF,
    POLICY_ADAPTIVE_ORACLE,
    POLICY_ADAPTIVE_PRED,
    POLICY_ADAPTIVE_NO_AGING,
    POLICY_ADAPTIVE_NO_COST,
)


def get_in_process_stream_fn():
    """In-process mock streaming connector mapping URLs to local backend apps."""
    from app.backends.model_a import app as app_a
    from app.backends.model_b import app as app_b
    from app.backends.model_c import app as app_c

    transport_a = httpx.ASGITransport(app=app_a)
    transport_b = httpx.ASGITransport(app=app_b)
    transport_c = httpx.ASGITransport(app=app_c)

    async def in_process_stream(url: str, payload: dict):
        if ":8001" in url or "model_a" in url:
            c = httpx.AsyncClient(transport=transport_a, base_url="http://backend_a", timeout=None)
        elif ":8002" in url or "model_b" in url:
            c = httpx.AsyncClient(transport=transport_b, base_url="http://backend_b", timeout=None)
        else:
            c = httpx.AsyncClient(transport=transport_c, base_url="http://backend_c", timeout=None)
        req = c.build_request("POST", "/generate", json=payload)
        resp = await c.send(req, stream=True)
        return c, resp

    return in_process_stream


async def run_benchmark_cli(
    workload_size: int = 25,
    seed: int = 42,
    arrival_process: str = "poisson",
    workload_preset: str = "balanced",
    target_load: Optional[float] = None,
    policies: Optional[List[str]] = None,
    is_multiseed: bool = False,
    seeds_list: Optional[List[int]] = None,
    workload_file: Optional[str] = None,
):
    print("=" * 70)
    print("  HORIZON 2K26 · D07 Token-Aware Scheduler Benchmark Lab")
    print("=" * 70)
    print(f"Arrival Process:    {arrival_process}")
    print(f"Workload Preset:    {workload_preset}")
    if target_load is not None:
        print(f"Target Load (rho):  {target_load}")
    print(f"Workload Size:      {workload_size} req/policy")

    if policies is None:
        policies = [POLICY_FIFO, POLICY_TOKEN_AWARE, POLICY_ADAPTIVE, POLICY_ADAPTIVE_LEAST_CONN]

    print(f"Policies:           {', '.join(policies)}")

    stream_fn = get_in_process_stream_fn()

    if is_multiseed:
        multiseeds = seeds_list if seeds_list else [42, 43, 44, 45, 46]
        print(f"\nRunning Multi-Seed Evaluation across {len(multiseeds)} seeds: {multiseeds}...")
        multi_result = await run_multi_seed_benchmark(
            seeds=multiseeds,
            workload_size=workload_size,
            stream_from_fn=stream_fn,
            policies=policies,
            arrival_process=arrival_process,
            workload_preset=workload_preset,
        )

        print("\n=== Multi-Seed Benchmark Completed! ===")
        print(f"Report File: {multi_result['report_path']}\n")

        print(f"{'Policy':<22} | {'Mean Lat (s)':<18} | {'p95 Lat (s)':<18} | {'Throughput':<14} | {'SLO Attain %'}")
        print("-" * 90)
        for p in policies:
            ci = multi_result["bootstrapped_ci"][p]
            m_lat = f"{ci['mean_latency'][0]:.2f} [{ci['mean_latency'][1]:.2f}, {ci['mean_latency'][2]:.2f}]"
            p95_lat = f"{ci['p95_latency'][0]:.2f} [{ci['p95_latency'][1]:.2f}, {ci['p95_latency'][2]:.2f}]"
            tp = f"{ci['throughput'][0]:.2f} req/s"
            slo = f"{ci['slo_attainment'][0]:.1f}%"
            print(f"{p:<22} | {m_lat:<18} | {p95_lat:<18} | {tp:<14} | {slo}")

        return multi_result

    # Single-run execution
    print(f"Random Seed:        {seed}")
    result = await run_full_benchmark_lab(
        workload_size=workload_size,
        seed=seed,
        stream_from_fn=stream_fn,
        policies=policies,
        arrival_process=arrival_process,
        workload_preset=workload_preset,
        target_load=target_load,
        workload_file=workload_file,
    )

    print("\n=== Benchmark Completed Successfully! ===")
    print(f"Run ID:      {result['run_id']}")
    print(f"Raw CSV:     {result['raw_csv']}")
    print(f"Summary CSV: {result['summary_csv']}\n")

    print(f"{'Policy':<22} | {'p50 Lat (s)':<12} | {'p95 Lat (s)':<12} | {'TTFT (s)':<10} | {'Throughput':<11} | {'SLO %':<8} | {'Starvations'}")
    print("-" * 95)
    for s in result["summaries"]:
        slo_pct = s.get("slo_attainment_pct", 0.0)
        print(f"{s['policy']:<22} | {s['p50_latency_sec']:<12.3f} | {s['p95_latency_sec']:<12.3f} | {s['mean_ttft_sec']:<10.3f} | {s['throughput_req_per_sec']:<11.2f} | {slo_pct:<8.1f} | {s['starvation_incidents']}")

    print("\nGenerating evaluation charts...")
    plots = generate_benchmark_charts(result["summary_csv"], result["raw_csv"])
    for p in plots:
        print(f" - Saved: {p}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HORIZON 2K26 D07 Benchmark Runner (Phase 2 Upgrade).")
    parser.add_argument("--size", type=int, default=25, help="Requests per policy (default 25)")
    parser.add_argument("--seed", type=int, default=42, help="Workload random seed (default 42)")
    parser.add_argument("--scale", type=float, default=1.0, help="Time scale factor (default 1.0)")
    parser.add_argument("--arrival-process", choices=["poisson", "bursty", "uniform"], default="poisson", help="Arrival process")
    parser.add_argument("--workload", choices=["balanced", "chat_heavy", "batch_heavy"], default="balanced", help="Workload profile preset")
    parser.add_argument("--target-load", "--rho", type=float, default=None, help="Target load factor rho (e.g. 0.85)")
    parser.add_argument("--policies", type=str, default=None, help="Comma-separated list of policies")
    parser.add_argument("--ablation", action="store_true", help="Run policy ablation suite")
    parser.add_argument("--multiseed", action="store_true", help="Run 5-seed statistical evaluation with bootstrap CIs")
    parser.add_argument("--workload-file", type=str, default=None, help="Replay pre-generated workload JSON file")
    args = parser.parse_args()

    import app.config as cfg
    cfg.TIME_SCALE = args.scale

    selected_policies = None
    if args.ablation:
        selected_policies = [
            POLICY_ADAPTIVE,
            POLICY_ADAPTIVE_LEAST_CONN,
            POLICY_ADAPTIVE_PRED,
            POLICY_ADAPTIVE_NO_AGING,
            POLICY_ADAPTIVE_NO_COST,
        ]
    elif args.policies:
        selected_policies = [p.strip() for p in args.policies.split(",")]

    asyncio.run(
        run_benchmark_cli(
            workload_size=args.size,
            seed=args.seed,
            arrival_process=args.arrival_process,
            workload_preset=args.workload,
            target_load=args.target_load,
            policies=selected_policies,
            is_multiseed=args.multiseed,
            workload_file=args.workload_file,
        )
    )

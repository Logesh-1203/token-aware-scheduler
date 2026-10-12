"""FastAPI router: /metrics, /metrics/recent, /metrics/stream (SSE), /metrics/export.csv, /dashboard.

    store, hub = MetricsStore(), MetricsHub()
    app.include_router(build_router(store, hub))
"""
import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

DASHBOARD = Path(__file__).with_name("dashboard.html")


def build_router(store, hub) -> APIRouter:
    router = APIRouter()

    @router.get("/metrics")
    def metrics():
        res = store.summary()
        try:
            from app.gateway.scheduler import get_metrics
            res = {**get_metrics(), **res}
        except Exception:
            pass
        return res

    @router.get("/metrics/recent")
    def recent(limit: int = Query(50, ge=1, le=1000), model: Optional[str] = None,
               mode: Optional[str] = None, backend: Optional[str] = None,
               policy: Optional[str] = None, status: Optional[str] = None):
        return {"items": store.recent(limit, model=model, mode=mode, backend=backend,
                                      policy=policy, status=status)}

    @router.get("/metrics/export.csv")
    def export_csv():
        return Response(store.export_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="horizon_metrics.csv"'})

    @router.get("/metrics/stream")
    async def metrics_stream(request: Request):
        async def events():
            q = hub.subscribe()
            try:
                yield "retry: 2000\n\n"
                yield f"data: {json.dumps({'event': 'hello'})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.get("/dashboard")
    def dashboard():
        return FileResponse(DASHBOARD, media_type="text/html")

    @router.get("/scheduler/status")
    def scheduler_status():
        try:
            from app.gateway.scheduler import get_detailed_stats, get_backends_status, get_metrics
            from app.gateway.traces import trace_manager
            return {
                "metrics": get_metrics(),
                "backends": get_backends_status(),
                "details": get_detailed_stats(),
                "recent_traces": trace_manager.get_traces(limit=15),
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @router.post("/benchmark/compare")
    async def benchmark_compare(payload: dict):
        from app.gateway.benchmark import run_side_by_side_benchmark
        baseline = payload.get("baseline_policy", "fifo")
        adaptive = payload.get("adaptive_policy", "adaptive")
        size = int(payload.get("workload_size", 10))
        preset = payload.get("workload_preset", "balanced")
        mode = payload.get("mode", "SIMULATED")
        seed = int(payload.get("seed", 42))
        warmup = int(payload.get("warmup_count", 1))
        conc = int(payload.get("concurrency", 1))

        # Check in-process connector
        from app.gateway.router import _stream_from_backend
        try:
            res = await run_side_by_side_benchmark(
                baseline_policy=baseline,
                adaptive_policy=adaptive,
                workload_size=size,
                workload_preset=preset,
                mode=mode,
                seed=seed,
                warmup_count=warmup,
                concurrency=conc,
                stream_from_fn=_stream_from_backend,
                metrics_store=store,
            )
            # Notify dashboard via SSE
            hub.publish({"event": "benchmark_completed", "run_id": res["run_id"], "mode": mode})
            return res
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {exc}")

    @router.get("/benchmark/history")
    def benchmark_history(limit: int = Query(30, ge=1, le=100)):
        return {"runs": store.get_benchmark_runs(limit=limit)}

    @router.get("/benchmark/history/{run_id}")
    def benchmark_run_detail(run_id: str):
        from fastapi import HTTPException
        run = store.get_benchmark_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")
        return run

    @router.get("/benchmark/history/{run_id}/export.csv")
    def benchmark_run_export_csv(run_id: str):
        from fastapi import HTTPException
        csv_data = store.export_benchmark_csv(run_id)
        if not csv_data:
            raise HTTPException(status_code=404, detail=f"No raw records found for benchmark {run_id}")
        return Response(csv_data, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="benchmark_{run_id}.csv"'})

    # ---- SLO & Starvation Guard Endpoints ----------------------------------------
    @router.get("/scheduler/starvation-guard")
    def starvation_guard_status():
        from app.gateway.scheduler import get_starvation_guard_status
        return get_starvation_guard_status()

    @router.post("/scheduler/starvation-guard")
    def starvation_guard_update(settings: dict):
        from fastapi import HTTPException
        from app.gateway.starvation_guard import starvation_guard
        ok, msg = starvation_guard.update_config(settings)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        hub.publish({"event": "starvation_guard_updated", "config": starvation_guard.get_config()})
        return {"status": "ok", "message": msg, "config": starvation_guard.get_config()}

    @router.post("/scheduler/starvation-guard/reset")
    def starvation_guard_reset():
        from app.gateway.starvation_guard import starvation_guard
        starvation_guard.reset()
        hub.publish({"event": "starvation_guard_reset", "config": starvation_guard.get_config()})
        return {"status": "ok", "message": "Settings reset to defaults", "config": starvation_guard.get_config()}

    @router.post("/benchmark/starvation-compare")
    async def benchmark_starvation_compare(payload: dict):
        from fastapi import HTTPException
        from app.gateway.benchmark import run_starvation_guard_comparison
        from app.gateway.router import _stream_from_backend
        policy = payload.get("policy", "adaptive")
        size = int(payload.get("workload_size", 12))
        seed = int(payload.get("seed", 42))
        preset = payload.get("workload_preset", "heavy_batch")
        try:
            res = await run_starvation_guard_comparison(
                policy=policy,
                workload_size=size,
                seed=seed,
                workload_preset=preset,
                stream_from_fn=_stream_from_backend,
            )
            hub.publish({"event": "starvation_benchmark_completed", "policy": policy})
            return res
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Starvation benchmark failed: {exc}")

    return router

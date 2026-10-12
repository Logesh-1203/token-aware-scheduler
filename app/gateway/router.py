from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import httpx
import asyncio
import json
from typing import Dict, Any, Optional
import time

from ..schemas import InferenceRequest
from ..config import (
    BACKEND_URLS,
    POLICY_FIFO,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    POLICY_ADAPTIVE_LEAST_CONN,
)
from .scheduler import (
    _enqueue,
    _queue_lock,
    _metrics_increment,
    get_metrics,
    get_detailed_stats,
    get_backends_status,
    increment_active,
    decrement_active,
    dispatcher_loop,
    _queue,
    record_queue_wait,
    record_request_completed,
    record_request_failed,
    schedule_and_acquire,
    release_slot,
    reset_scheduler_state,
    check_admission_control,
)
from .health import health_manager
from .traces import trace_manager
from .benchmark import run_full_benchmark_lab
from .predictor import default_predictor

import uuid
from ..metrics_store import MetricsStore, MetricsHub
from ..metrics_routes import build_router
from ..instrument import instrument_stream

router = APIRouter()

metrics_store = MetricsStore()
metrics_hub = MetricsHub()
router.include_router(build_router(metrics_store, metrics_hub))

@router.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}

@router.get("/backends")
async def backends_status():
    async with httpx.AsyncClient() as client:
        statuses = {}
        for name, url in BACKEND_URLS.items():
            try:
                resp = await client.get(f"{url}/health", timeout=2.0)
                health_data = resp.json()
            except Exception as e:
                health_data = {"status": "unhealthy", "error": str(e)}
            
            sched_status = get_backends_status().get(name, {})
            statuses[name] = {
                **health_data,
                **sched_status,
                "circuit_state": health_manager.get_state(name),
            }
    return statuses

async def _stream_from_backend(backend_url: str, payload: dict) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Create a client, send a streaming request, and return both client and response.
    Caller must close the client after consuming the stream.
    """
    client = httpx.AsyncClient(timeout=None)
    request = client.build_request("POST", f"{backend_url}/generate", json=payload)
    resp = await client.send(request, stream=True)
    return client, resp

async def _stream_generator(
    client: httpx.AsyncClient,
    resp: httpx.Response,
    backend_name: str,
    req_dict: Optional[dict] = None,
    wait_sec: float = 0.0,
    sched_metadata: Optional[dict] = None,
):
    """Yield response chunks, update metrics, trace decisions, and clean up.
    On normal completion increments total_completed.
    On any exception increments total_failed.
    Always decrements active request counter and closes client.
    """
    t_start = time.monotonic()
    first_token_time = None
    tokens_yielded = 0
    enqueued_at = req_dict.get("enqueued_at", t_start) if req_dict else t_start
    profile = req_dict.get("profile", "normal") if req_dict else "normal"
    policy = req_dict.get("policy", "fifo") if req_dict else "fifo"
    rid = str(req_dict.get("request_id") or uuid.uuid4().hex) if req_dict else str(uuid.uuid4().hex)
    use_real = bool(req_dict.get("use_real_model", False)) if req_dict else False
    mode = "REAL" if use_real else "SIMULATED"
    model_name = "qwen3:4b-q4_K_M" if use_real else backend_name

    sched_meta = sched_metadata or {}
    est_input = (req_dict.get("estimated_input_tokens") if req_dict else None) or sched_meta.get("estimated_input_tokens")
    max_out = (req_dict.get("max_output_tokens") if req_dict else None) or sched_meta.get("max_output_tokens")
    r_score = sched_meta.get("routing_score")
    if r_score is None and sched_meta.get("score_breakdown"):
        r_score = sched_meta["score_breakdown"].get("total_score")
    r_expl = sched_meta.get("routing_explanation") or ""

    try:
        metrics_store.start(
            rid,
            prompt=req_dict.get("prompt", "") if req_dict else "",
            model=model_name,
            mode=mode,
            backend=backend_name,
            policy=policy,
            profile=profile,
            estimated_input_tokens=est_input,
            max_output_tokens=max_out,
            routing_score=r_score,
            routing_explanation=r_expl,
            queue_wait_s=round(wait_sec, 4),
        )
        metrics_hub.publish({
            "event": "start",
            "request_id": rid,
            "mode": mode,
            "backend": backend_name,
            "policy": policy,
            "routing_score": r_score,
            "routing_explanation": r_expl,
        })
    except Exception:
        pass

    meta_dict = {}

    try:
        async for chunk in resp.aiter_text():
            if first_token_time is None:
                first_token_time = time.monotonic()
            tokens_yielded += 1

            try:
                new_lines = []
                has_mod = False
                for line in chunk.splitlines():
                    if not line.strip():
                        continue
                    c_ev = json.loads(line)
                    if isinstance(c_ev, dict) and c_ev.get("type") in ("done", "error"):
                        meta_dict.update(c_ev)
                        if use_real and c_ev.get("type") == "done":
                            c_ev["server_latency_s"] = round(time.monotonic() - enqueued_at, 4)
                            c_ev["server_ttft_s"] = round(first_token_time - enqueued_at, 4) if first_token_time else None
                            c_ev["status"] = "SUCCESS"
                            new_lines.append(json.dumps(c_ev, ensure_ascii=False))
                            has_mod = True
                            continue
                    new_lines.append(line)
                if has_mod:
                    chunk = "\n".join(new_lines) + "\n"
            except Exception:
                pass

            yield chunk

        t_end = time.monotonic()
        completion_latency = t_end - enqueued_at
        ttft_sec = (first_token_time - enqueued_at) if first_token_time else completion_latency

        record_request_completed(profile, policy, wait_sec, completion_latency, ttft_sec)
        health_manager.record_success(backend_name, completion_latency, tokens_yielded)
        default_predictor.observe(profile, req_dict.get("prompt", "") if req_dict else "", tokens_yielded)

        # Record in metrics_store and metrics_hub
        try:
            status = "SUCCESS" if meta_dict.get("type") != "error" else "FAILED"
            p_tokens = meta_dict.get("prompt_tokens")
            o_tokens = meta_dict.get("output_tokens") or meta_dict.get("tokens_generated") or tokens_yielded
            tps = meta_dict.get("tokens_per_second")
            if tps is None and completion_latency > 0 and o_tokens:
                tps = o_tokens / completion_latency

            metrics_store.finish(
                rid,
                status=status,
                latency_s=round(completion_latency, 4),
                ttft_s=round(ttft_sec, 4) if ttft_sec else None,
                prompt_tokens=p_tokens,
                output_tokens=o_tokens,
                tokens_per_second=round(tps, 2) if tps else None,
                prompt_eval_s=meta_dict.get("prompt_eval_s"),
                load_s=meta_dict.get("load_s"),
                ollama_total_s=meta_dict.get("ollama_total_s"),
                retries=req_dict.get("retries", 0) if req_dict else 0,
                error=meta_dict.get("error"),
            )
            metrics_hub.publish({"event": "finish", "request_id": rid, "status": status})
        except Exception:
            pass

        if req_dict:
            sched_meta = sched_metadata or {}
            explanation = trace_manager.generate_explanation(
                policy=policy,
                profile=profile,
                selected_backend=backend_name,
                queue_wait=wait_sec,
                score_breakdown=sched_meta.get("score_breakdown", {}),
                candidate_estimates=sched_meta.get("candidate_estimates", {}),
                rejected_reasons=sched_meta.get("rejected_reasons", {}),
                success=True,
            )
            trace_manager.record_trace({
                "request_id": req_dict.get("request_id"),
                "policy": policy,
                "profile": profile,
                "max_output_tokens": req_dict.get("max_output_tokens"),
                "selected_backend": backend_name,
                "queue_wait": round(wait_sec, 4),
                "ttft": round(ttft_sec, 4),
                "completion_latency": round(completion_latency, 4),
                "tokens_emitted": tokens_yielded,
                "final_status": "completed",
                "explanation": explanation,
                "candidate_estimates": sched_meta.get("candidate_estimates", {}),
            })
    except Exception as ex:
        record_request_failed(profile, policy)
        health_manager.record_failure(backend_name, str(ex))
        try:
            metrics_store.finish(
                rid,
                status="FAILED",
                latency_s=round(time.monotonic() - enqueued_at, 4),
                ttft_s=round(first_token_time - enqueued_at, 4) if first_token_time else None,
                error=str(ex),
            )
            metrics_hub.publish({"event": "finish", "request_id": rid, "status": "FAILED"})
        except Exception:
            pass
        if req_dict:
            trace_manager.record_trace({
                "request_id": req_dict.get("request_id"),
                "policy": policy,
                "profile": profile,
                "selected_backend": backend_name,
                "queue_wait": round(wait_sec, 4),
                "final_status": "failed",
                "error": str(ex),
            })
        raise
    finally:
        await client.aclose()
        await release_slot(backend_name)

@router.post("/infer", response_class=StreamingResponse)
async def infer(request: InferenceRequest):
    """Stream inference tokens via chunked NDJSON (application/x-ndjson).
    Supports simulated backends and live Ollama Qwen3 via use_real_model=true.
    """
    req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()

    # Admission control check (optional, off by default)
    if not check_admission_control(req_dict.get("profile", "normal")):
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": "2"},
            detail="Admission control: queue overloaded, request shed",
        )

    use_real = req_dict.get("use_real_model", False)

    payload = {
        "prompt": req_dict["prompt"],
        "max_output_tokens": req_dict["max_output_tokens"],
        "request_id": req_dict.get("request_id"),
        "simulate_failure": req_dict.get("simulate_failure", "none"),
        "use_real_model": use_real,
    }
    if "sim_true_output_tokens" in req_dict and req_dict["sim_true_output_tokens"] is not None:
        payload["sim_true_output_tokens"] = req_dict["sim_true_output_tokens"]

    # 1. Schedule request and acquire backend slot
    # In real-model mode, route through scheduler targeting Model A (Ollama-backed)
    target_urls = {"model_a": BACKEND_URLS["model_a"]} if use_real else BACKEND_URLS
    backend_name, backend_url, wait_sec, sched_meta = await schedule_and_acquire(req_dict, backend_urls=target_urls)

    try:
        client, resp = await _stream_from_backend(backend_url, payload)
        if getattr(resp, "status_code", 200) >= 400:
            err_body = await resp.aread() if hasattr(resp, "aread") else b""
            raise HTTPException(
                status_code=getattr(resp, "status_code", 502),
                detail=f"Backend returned HTTP error: {err_body.decode('utf-8', errors='ignore')}"
            )
        return StreamingResponse(
            _stream_generator(client, resp, backend_name, req_dict, wait_sec, sched_meta),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        # 2. Failure happened BEFORE any tokens were sent to client!
        await release_slot(backend_name)
        health_manager.record_failure(backend_name, str(e))

        # Check single-retry on alternate healthy backend for simulated requests
        if not use_real:
            healthy_candidates = [
                b for b in BACKEND_URLS.keys()
                if b != backend_name and health_manager.is_available(b)
            ]

            if healthy_candidates:
                _metrics_increment("total_retried")
                req_dict["fallback_occurred"] = True
                sched_meta["fallback_occurred"] = True
                retry_name = healthy_candidates[0]
                retry_url = BACKEND_URLS[retry_name]
                increment_active(retry_name)

                try:
                    retry_client, retry_resp = await _stream_from_backend(retry_url, payload)
                    if getattr(retry_resp, "status_code", 200) >= 400:
                        raise HTTPException(status_code=getattr(retry_resp, "status_code", 502), detail="Retry backend returned HTTP error")
                    return StreamingResponse(
                        _stream_generator(retry_client, retry_resp, retry_name, req_dict, wait_sec, sched_meta),
                        media_type="application/x-ndjson"
                    )
                except Exception as e2:
                    await release_slot(retry_name)
                    health_manager.record_failure(retry_name, str(e2))
                    record_request_failed(req_dict.get("profile", "normal"), req_dict.get("policy", "fifo"))
                    raise HTTPException(status_code=502, detail=f"Primary ({backend_name}) and retry ({retry_name}) failed: {e2}")

        record_request_failed(req_dict.get("profile", "normal"), req_dict.get("policy", "fifo"))
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/chat")
async def chat(request: InferenceRequest):
    """Convenience chat endpoint returning standard JSON response for easy Swagger UI testing.
    Internally consumes the streaming tokens and returns the complete text.
    """
    stream_response = await infer(request)
    full_text = []
    thinking_text = []
    tokens_generated = 0
    model_name = "qwen3"

    async for chunk in stream_response.body_iterator:
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("type") == "token":
                    full_text.append(item.get("text", ""))
                elif item.get("type") == "thinking":
                    thinking_text.append(item.get("text", ""))
                elif item.get("type") == "done":
                    tokens_generated = item.get("tokens_generated", len(full_text))
                    model_name = item.get("model", model_name)
                elif item.get("type") == "error":
                    raise HTTPException(status_code=502, detail=item.get("error", "Stream error"))
            except json.JSONDecodeError:
                continue

    reply = "".join(full_text)
    if not reply.strip() and thinking_text:
        reply = "".join(thinking_text)

    return {
        "request_id": request.request_id,
        "prompt": request.prompt,
        "response": reply,
        "model": model_name,
        "policy": request.policy.value if hasattr(request.policy, "value") else str(request.policy),
        "tokens_generated": tokens_generated or len(full_text) or len(thinking_text),
        "use_real_model": request.use_real_model,
    }



@router.get("/stats")
async def detailed_stats():
    return get_detailed_stats()

@router.get("/queue")
async def queue_view():
    async with _queue_lock:
        pending = list(_queue)
    return {"length": len(pending), "requests": pending}

@router.get("/traces")
async def get_all_traces(limit: int = 50):
    return trace_manager.get_traces(limit=limit)

@router.get("/traces/{request_id}")
async def get_single_trace(request_id: str):
    trace = trace_manager.get_trace(request_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace for request {request_id} not found")
    return trace

@router.post("/admin/reset")
async def admin_reset():
    reset_scheduler_state()
    return {"status": "ok", "message": "All queues, metrics, circuit breakers, and traces have been reset."}

@router.post("/benchmark")
async def run_benchmark(workload_size: int = 25, seed: int = 42):
    try:
        report = await run_full_benchmark_lab(
            workload_size=workload_size,
            seed=seed,
            stream_from_fn=_stream_from_backend,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")

@router.on_event("startup")
async def start_dispatcher():
    asyncio.create_task(dispatcher_loop())

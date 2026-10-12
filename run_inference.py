#!/usr/bin/env python
"""HORIZON 2K26 - Terminal Inference Client.

Every prompt goes through the FastAPI gateway (POST /infer), so the scheduler routes it and the
metrics pipeline records it for the dashboard. This client never calls Ollama directly.
Standard library only.

    .\\.venv\\Scripts\\python.exe run_inference.py                    # interactive, REAL model
    .\\.venv\\Scripts\\python.exe run_inference.py --simulated        # simulated backend (labelled SIMULATED)
    .\\.venv\\Scripts\\python.exe run_inference.py --bench 5 --csv bench.csv --prompt "Explain TTFT briefly"
"""
import argparse
import csv
import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.metrics_store import aggregate  # noqa: E402

GATEWAY = os.environ.get("HORIZON_GATEWAY", "http://127.0.0.1:9000")
DEFAULT_PROMPTS = [
    "Explain token-aware scheduling in two sentences.",
    "What is time to first token (TTFT) in LLM inference?",
    "Give three tips to reduce tail latency in a request scheduler.",
]
_MOCK_RE = re.compile(r"(token\d+)+")


def build_payload(prompt, *, request_id, real, max_output_tokens, field="prompt", profile=None):
    """The ONE place that defines the /infer request body. If your gateway's request model uses
    different field names (e.g. 'text' instead of 'prompt'), change them here or pass --prompt-field."""
    payload = {"request_id": request_id, field: prompt, "use_real_model": bool(real),
               "max_output_tokens": max_output_tokens}
    if profile:
        payload["profile"] = profile
    return payload


def run_one(prompt, *, gateway=GATEWAY, real=True, max_tokens=256, field="prompt", profile=None,
            timeout=600.0, echo=True, out=None):
    """Send one prompt through the gateway, stream the answer, return a result dict."""
    out = out or sys.stdout
    rid = uuid.uuid4().hex
    body = json.dumps(build_payload(prompt, request_id=rid, real=real, max_output_tokens=max_tokens,
                                    field=field, profile=profile)).encode("utf-8")
    req = urllib.request.Request(
        gateway.rstrip("/") + "/infer", data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson", "X-Request-ID": rid})
    res = {"request_id": rid, "mode": "REAL" if real else "SIMULATED", "status": "FAILED", "error": None,
           "text": "", "client_latency_s": None, "client_ttft_s": None, "done": {}}
    parts, done, err_ev = [], None, None
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    res["error"] = f"gateway sent a line that is not JSON: {line[:100]!r}"
                    break
                kind = ev.get("type") if isinstance(ev, dict) else None
                if kind == "token":
                    text = ev.get("text") or ""
                    if text and res["client_ttft_s"] is None:
                        res["client_ttft_s"] = time.perf_counter() - t0
                    parts.append(text)
                    if echo:
                        out.write(text)
                        out.flush()
                elif kind == "done":
                    done = ev
                elif kind == "error":
                    err_ev = ev
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        res["error"] = f"gateway returned HTTP {exc.code}: {detail}"
    except urllib.error.URLError as exc:
        res["error"] = f"cannot reach the gateway at {gateway}: {exc.reason}. Is it running on that port?"
    except (socket.timeout, TimeoutError):
        res["error"] = f"no response from the gateway within {timeout:.0f}s"
    except (http.client.HTTPException, OSError) as exc:
        res["error"] = f"connection to the gateway broke mid-stream: {type(exc).__name__}: {exc}"
    res["client_latency_s"] = time.perf_counter() - t0
    res["text"] = "".join(parts)

    if res["error"] is None:
        if err_ev is not None:
            res["error"] = f"{err_ev.get('kind', 'error')}: {err_ev.get('error', 'unknown error')}"
            res["done"] = err_ev
        elif done is None:
            res["error"] = "the stream ended without a final 'done' event"
        else:
            res["done"] = done
            if real and (done.get("mode") != "REAL" or _MOCK_RE.fullmatch(res["text"].strip() or "x")):
                res["error"] = "REAL mode was requested but a mock/simulated response was received"
            else:
                res["status"] = "SUCCESS"
    elif err_ev is not None:
        res["done"] = err_ev
    return res


def _f(v, digits=3, unit=""):
    return "N/A" if v is None else f"{v:.{digits}f}{unit}"


def print_summary(res, out=None):
    out = out or sys.stdout
    d = res["done"] or {}
    tps = d.get("tokens_per_second")
    lines = [
        f"Request ID: {res['request_id']}",
        f"Model: {d.get('model') or 'N/A'}",
        f"Execution mode: {res['mode']}",
        f"End-to-end Latency (client): {_f(res['client_latency_s'], 3, ' seconds')}",
        f"Latency (gateway): {_f(d.get('server_latency_s'), 3, ' seconds')}",
        f"TTFT (client): {_f(res['client_ttft_s'], 3, ' seconds')}",
        f"TTFT (gateway): {_f(d.get('server_ttft_s'), 3, ' seconds')}",
        f"Prompt Tokens: {d.get('prompt_tokens', 'N/A') if d.get('prompt_tokens') is not None else 'N/A'}",
        f"Output Tokens: {d.get('output_tokens') if d.get('output_tokens') is not None else 'N/A'}",
        f"Generation Throughput (model-only): {_f(tps, 2, ' tokens/second')}",
        f"Status: {res['status']}",
    ]
    if res["error"]:
        lines.append(f"Error: {res['error']}")
    out.write("\n" + "\n".join(lines) + "\n")
    out.flush()


def check_gateway(gateway):
    try:
        with urllib.request.urlopen(gateway.rstrip("/") + "/health", timeout=3) as r:
            return True, f"gateway OK (HTTP {r.status})"
    except Exception as exc:  # noqa: BLE001 - just reporting
        return False, f"gateway not reachable at {gateway}: {getattr(exc, 'reason', exc)}"


def interactive(args):
    ok, msg = check_gateway(args.gateway)
    print("=" * 60)
    print("HORIZON 2K26 \u2014 Terminal Inference Client")
    print(f"Gateway: {args.gateway}   Mode: {'SIMULATED' if args.simulated else 'REAL'}")
    print(("\u2713 " if ok else "! WARNING: ") + msg)
    print("Commands: /quit  /real  /sim  /help      Dashboard: " + args.gateway.rstrip("/") + "/dashboard")
    print("=" * 60)
    real = not args.simulated
    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        cmd = prompt.lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/help":
            print("Type a question and press Enter. /real and /sim switch execution mode. /quit exits.")
            continue
        if cmd in ("/real", "/sim"):
            real = cmd == "/real"
            print(f"Execution mode is now {'REAL' if real else 'SIMULATED'}.")
            continue
        sys.stdout.write("Model response: ")
        sys.stdout.flush()
        res = run_one(prompt, gateway=args.gateway, real=real, max_tokens=args.max_tokens,
                      field=args.prompt_field, profile=args.profile, timeout=args.timeout)
        print_summary(res)
    print("Bye.")


def bench(args):
    prompts = [args.prompt] * args.bench if args.prompt else [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(args.bench)]
    mode = "SIMULATED" if args.simulated else "REAL"
    print(f"Benchmark: {args.bench} request(s) through {args.gateway}, mode={mode}")
    runs = []
    for i, p in enumerate(prompts, 1):
        r = run_one(p, gateway=args.gateway, real=not args.simulated, max_tokens=args.max_tokens,
                    field=args.prompt_field, profile=args.profile, timeout=args.timeout, echo=False)
        d = r["done"] or {}
        runs.append({"run": i, "request_id": r["request_id"], "mode": r["mode"], "status": r["status"],
                     "latency_s": r["client_latency_s"], "ttft_s": r["client_ttft_s"],
                     "prompt_tokens": d.get("prompt_tokens"), "output_tokens": d.get("output_tokens"),
                     "tokens_per_second": d.get("tokens_per_second"), "error": r["error"]})
        print(f"  [{i}/{args.bench}] {r['status']:<7} latency={_f(r['client_latency_s'], 2, 's')} "
              f"ttft={_f(r['client_ttft_s'], 2, 's')} out_tokens={d.get('output_tokens', 'N/A')} "
              f"tok/s={_f(d.get('tokens_per_second'), 2)}" + (f"  ERROR: {r['error']}" if r["error"] else ""))
    a = aggregate(runs)
    print(f"\nSummary (mode={mode}; latency/TTFT measured client-side; tokens & tok/s from Ollama metadata; p95 = nearest-rank)")
    print(f"  Total requests: {a['total']}   Successful: {a['success']}   Failed: {a['failed']}   "
          f"Success rate: {_f(a['success_rate'] and a['success_rate'] * 100, 1, '%') if a['total'] else 'N/A'}")
    print(f"  Latency (n={a['n_latency']}):  avg={_f(a['avg_latency_s'], 3)}s  median={_f(a['median_latency_s'], 3)}s  p95={_f(a['p95_latency_s'], 3)}s")
    print(f"  TTFT (n={a['n_ttft']}):  avg={_f(a['avg_ttft_s'], 3)}s")
    print(f"  Output tokens (n={a['n_output_tokens']}):  avg={_f(a['avg_output_tokens'], 1)}")
    print(f"  Throughput (n={a['n_tokens_per_second']}):  avg={_f(a['avg_tokens_per_second'], 2)} tokens/s")
    print("  Successful-only averages; failed requests are counted separately above.")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)
        print(f"  Per-request CSV written to {args.csv}")
    return 0 if a["failed"] == 0 else 1


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Tamil and other non-ASCII text on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="HORIZON 2K26 terminal inference client (via the gateway)")
    ap.add_argument("--gateway", default=GATEWAY)
    ap.add_argument("--simulated", action="store_true", help="use the simulated backend path (results are labelled SIMULATED)")
    ap.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    ap.add_argument("--prompt-field", default="prompt", dest="prompt_field", help="name of the prompt field in /infer's request body")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="send N requests and print aggregate stats")
    ap.add_argument("--prompt", default=None, help="prompt to use for every --bench request")
    ap.add_argument("--csv", default=None, help="write per-request benchmark rows to this CSV file")
    args = ap.parse_args(argv)
    if args.bench > 0:
        return bench(args)
    interactive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

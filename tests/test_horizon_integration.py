"""Offline tests for the D07 real-inference + live-metrics pipeline (unittest; pytest runs it too).

These use controlled fakes. They prove the logic, NOT that your installed Qwen3 works:
that is what tests/manual_real_model_check.py is for.
"""
import asyncio
import http.client
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import run_inference as cli  # noqa: E402
from app.instrument import instrument_stream  # noqa: E402
from app.metrics_store import MetricsHub, MetricsStore, aggregate, percentile  # noqa: E402
from app.ollama_stream import (build_payload, final_metadata, parse_ollama_lines,  # noqa: E402
                               resolve_endpoint, tokens_per_second)

NS = 1_000_000_000
MODEL = "qwen3:4b-q4_K_M"


def chat(content, **extra):
    return json.dumps({"message": {"role": "assistant", "content": content}, "done": False, **extra})


FINAL = json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                    "prompt_eval_count": 11, "eval_count": 40, "eval_duration": 8 * NS,
                    "prompt_eval_duration": NS // 3, "load_duration": 3 * NS // 2, "total_duration": 10 * NS})


async def alines(lines, delay=0.0):
    for line in lines:
        if delay:
            await asyncio.sleep(delay)
        yield line


async def events(agen):
    return [x if isinstance(x, dict) else json.loads(x) async for x in agen]


def run(coro):
    return asyncio.run(coro)


class TestMath(unittest.TestCase):
    def test_tokens_per_second_uses_nanoseconds(self):
        self.assertAlmostEqual(tokens_per_second(40, 8 * NS), 5.0)
        self.assertAlmostEqual(tokens_per_second(100, NS // 2), 200.0)

    def test_tokens_per_second_safe_on_bad_input(self):
        for c, d in [(0, NS), (10, 0), (10, -5), (None, NS), (10, None), ("x", NS)]:
            self.assertIsNone(tokens_per_second(c, d))

    def test_final_metadata(self):
        m = final_metadata(json.loads(FINAL))
        self.assertEqual((m["prompt_tokens"], m["output_tokens"]), (11, 40))
        self.assertAlmostEqual(m["tokens_per_second"], 5.0)
        self.assertAlmostEqual(m["load_s"], 1.5)
        self.assertAlmostEqual(m["ollama_total_s"], 10.0)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4, 100], 50), 3)
        self.assertEqual(percentile([1, 2, 3, 4, 100], 95), 100)
        self.assertIsNone(percentile([], 95))

    def test_endpoint_and_payload(self):
        self.assertEqual(resolve_endpoint("http://127.0.0.1:11434"), "http://127.0.0.1:11434/api/chat")
        p = build_payload("hi", model=MODEL, url="http://x/api/chat", max_output_tokens=64, think=False)
        self.assertEqual(p["options"]["num_predict"], 64)
        self.assertEqual(p["messages"][0]["content"], "hi")
        self.assertIs(p["think"], False)
        g = build_payload("hi", model=MODEL, url="http://x/api/generate", max_output_tokens=7)
        self.assertEqual(g["prompt"], "hi")
        self.assertNotIn("messages", g)


class TestOllamaParsing(unittest.TestCase):
    def test_chat_stream_uses_final_metadata_not_chunk_count(self):
        evs = run(events(parse_ollama_lines(alines([chat("Hel"), chat("lo wor"), chat("ld"), FINAL]), chat=True)))
        self.assertEqual("".join(e["text"] for e in evs if e["type"] == "token"), "Hello world")
        done = evs[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["mode"], "REAL")
        self.assertEqual(done["output_tokens"], 40)  # 3 chunks != 40 tokens
        self.assertAlmostEqual(done["tokens_per_second"], 5.0)

    def test_generate_schema(self):
        lines = [json.dumps({"response": "Hi", "done": False}),
                 json.dumps({"response": "", "done": True, "eval_count": 2, "eval_duration": NS})]
        evs = run(events(parse_ollama_lines(alines(lines), chat=False)))
        self.assertEqual(evs[0], {"type": "token", "text": "Hi"})
        self.assertEqual(evs[-1]["output_tokens"], 2)

    def test_thinking_is_separate_from_visible_tokens(self):
        lines = [json.dumps({"message": {"thinking": "hmm"}, "done": False}), chat("Answer"), FINAL]
        evs = run(events(parse_ollama_lines(alines(lines), chat=True)))
        self.assertEqual([e["type"] for e in evs], ["thinking", "token", "done"])

    def test_error_malformed_and_truncated(self):
        e = run(events(parse_ollama_lines(alines([json.dumps({"error": "model 'x' not found"})]))))
        self.assertEqual((e[0]["type"], e[0]["kind"]), ("error", "ollama_error"))
        e = run(events(parse_ollama_lines(alines(["not json"]))))
        self.assertEqual(e[0]["kind"], "malformed_response")
        e = run(events(parse_ollama_lines(alines([chat("partial")]))))
        self.assertEqual((e[0]["type"], e[-1]["kind"]), ("token", "truncated"))


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "m.db")
        self.store = MetricsStore(self.path)
        self.hub = MetricsHub()

    def tearDown(self):
        self.store._db.close()
        self.tmp.cleanup()

    def pipe(self, upstream, rid="r1", mode="REAL", **kw):
        return instrument_stream(upstream, store=self.store, hub=self.hub, request_id=rid, prompt="why?",
                                 model=MODEL, mode=mode, backend="model_a", policy="token_aware",
                                 profile="chat", **kw)

    def real_upstream(self, delay=0.0):
        return parse_ollama_lines(alines([chat("Hel"), chat("lo"), FINAL], delay), chat=True)


class TestStore(StoreCase):
    def test_persists_across_restart(self):
        self.store.start("a", prompt="p", model=MODEL, mode="REAL")
        self.store.finish("a", status="SUCCESS", latency_s=1.5, output_tokens=40)
        reopened = MetricsStore(self.path)
        row = reopened.recent(1)[0]
        self.assertEqual((row["status"], row["latency_s"], row["output_tokens"]), ("SUCCESS", 1.5, 40))
        reopened._db.close()

    def test_unfinished_requests_become_failed_after_restart(self):
        self.store.start("stuck", prompt="p", model=MODEL, mode="REAL")
        reopened = MetricsStore(self.path)
        row = reopened.recent(1)[0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIn("interrupted", row["error"])
        reopened._db.close()

    def test_finish_is_idempotent_no_double_count(self):
        self.store.start("a", prompt="p", model=MODEL, mode="REAL")
        self.assertIsNotNone(self.store.finish("a", status="SUCCESS", latency_s=1.0))
        self.assertIsNone(self.store.finish("a", status="FAILED", error="late"))
        s = self.store.summary()
        self.assertEqual((s["success"], s["failed"]), (1, 0))

    def test_duplicate_request_id_stays_unique(self):
        r1 = self.store.start("same", prompt="p", model=MODEL, mode="REAL")
        r2 = self.store.start("same", prompt="p", model=MODEL, mode="REAL")
        self.assertNotEqual(r1["request_id"], r2["request_id"])
        self.assertTrue(r2["request_id"].startswith("same"))

    def test_filters_and_csv(self):
        for rid, mode, status in [("a", "REAL", "SUCCESS"), ("b", "SIMULATED", "SUCCESS"), ("c", "REAL", "FAILED")]:
            self.store.start(rid, prompt="p", model=MODEL, mode=mode, backend="model_a")
            self.store.finish(rid, status=status, latency_s=1.0)
        self.assertEqual({r["request_id"] for r in self.store.recent(10, mode="REAL")}, {"a", "c"})
        self.assertEqual([r["request_id"] for r in self.store.recent(10, mode="REAL", status="FAILED")], ["c"])
        lines = self.store.export_csv().strip().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("request_id,created_at"))

    def test_summary_keeps_real_and_simulated_apart_and_excludes_failures(self):
        def add(rid, mode, status, lat, ttft=None):
            self.store.start(rid, prompt="p", model=MODEL, mode=mode)
            self.store.finish(rid, status=status, latency_s=lat, ttft_s=ttft, output_tokens=10, tokens_per_second=5.0)
        add("r1", "REAL", "SUCCESS", 2.0, 0.5)
        add("r2", "REAL", "SUCCESS", 4.0)           # TTFT unavailable
        add("r3", "REAL", "FAILED", 99.0)           # must not enter averages
        add("s1", "SIMULATED", "SUCCESS", 8.0, 4.0)
        s = self.store.summary()
        real = s["by_mode"]["REAL"]
        self.assertEqual((real["success"], real["failed"], real["n_ttft"]), (2, 1, 1))
        self.assertAlmostEqual(real["avg_latency_s"], 3.0)
        self.assertAlmostEqual(real["avg_ttft_s"], 0.5)
        self.assertAlmostEqual(s["by_mode"]["SIMULATED"]["avg_latency_s"], 8.0)
        self.assertAlmostEqual(s["success_rate"], 3 / 4)

    def test_empty_summary_has_no_invented_numbers(self):
        s = self.store.summary()
        self.assertIsNone(s["latest"])
        self.assertIsNone(s["success_rate"])
        self.assertIsNone(s["by_mode"]["REAL"]["avg_latency_s"])


class TestInstrument(StoreCase):
    def test_real_success_records_authoritative_metrics_and_ttft(self):
        out = run(events(self.pipe(self.real_upstream(delay=0.15))))
        done = out[-1]
        self.assertEqual((done["type"], done["status"], done["mode"]), ("done", "SUCCESS", "REAL"))
        row = self.store.recent(1)[0]
        self.assertEqual(row["status"], "SUCCESS")
        self.assertEqual((row["prompt_tokens"], row["output_tokens"]), (11, 40))
        self.assertAlmostEqual(row["tokens_per_second"], 5.0)
        self.assertGreaterEqual(row["ttft_s"], 0.15)                 # waited for the first token
        self.assertGreaterEqual(row["latency_s"], row["ttft_s"])
        self.assertAlmostEqual(row["load_s"], 1.5)
        self.assertEqual([r["request_id"] for r in self.store.recent(5)], ["r1"])  # recorded once

    def test_ttft_ignores_empty_chunks(self):
        up = parse_ollama_lines(alines([chat(""), chat(""), chat("x"), FINAL], 0.05), chat=True)
        run(events(self.pipe(up)))
        self.assertGreaterEqual(self.store.recent(1)[0]["ttft_s"], 0.05)

    def test_backend_error_event_is_failed_with_no_fake_numbers(self):
        up = alines([json.dumps({"type": "error", "kind": "ollama_unreachable", "error": "Cannot connect"})])
        out = run(events(self.pipe(up)))
        self.assertEqual(out[-1]["type"], "error")
        row = self.store.recent(1)[0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIn("ollama_unreachable", row["error"])
        self.assertIsNone(row["output_tokens"])
        self.assertIsNone(row["tokens_per_second"])

    def test_mock_output_in_real_mode_is_failed(self):
        sim = [json.dumps({"type": "token", "text": f"token{i}"}) for i in range(1, 6)]
        sim.append(json.dumps({"type": "done", "mode": "SIMULATED", "output_tokens": 5}))
        out = run(events(self.pipe(alines(sim), mode="REAL")))
        self.assertEqual(out[-1]["type"], "error")
        row = self.store.recent(1)[0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIn("invalid_real_response", row["error"])

    def test_simulated_mode_is_labelled_simulated_and_succeeds(self):
        sim = [json.dumps({"type": "token", "text": "token1"}),
               json.dumps({"type": "done", "mode": "SIMULATED", "output_tokens": 1})]
        out = run(events(self.pipe(alines(sim), mode="SIMULATED")))
        self.assertEqual((out[-1]["type"], out[-1]["mode"]), ("done", "SIMULATED"))
        self.assertEqual(self.store.recent(1)[0]["mode"], "SIMULATED")

    def test_backend_crash_midstream_is_recorded_and_reported(self):
        async def crashing():
            yield json.dumps({"type": "token", "text": "partial"})
            raise RuntimeError("backend exploded")
        out = run(events(self.pipe(crashing())))
        self.assertEqual([e["type"] for e in out], ["token", "error"])
        row = self.store.recent(1)[0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIn("backend exploded", row["error"])

    def test_stream_without_done_is_failed(self):
        run(events(self.pipe(alines([json.dumps({"type": "token", "text": "x"})]))))
        self.assertIn("truncated", self.store.recent(1)[0]["error"])

    def test_client_disconnect_is_recorded_as_failed(self):
        async def go():
            gen = self.pipe(self.real_upstream(delay=0.05))
            await gen.__anext__()          # client reads the first token ...
            await gen.aclose()             # ... then goes away
        run(go())
        row = self.store.recent(1)[0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIn("interrupted", row["error"])

    def test_live_events_published(self):
        q = self.hub.subscribe()
        run(events(self.pipe(self.real_upstream())))
        got = []
        while not q.empty():
            got.append(q.get_nowait()["event"])
        self.assertEqual(got, ["start", "finish"])

    def test_concurrent_requests_do_not_mix(self):
        def up(word, n):
            fin = json.loads(FINAL)
            fin["eval_count"] = n
            return parse_ollama_lines(alines([chat(word), json.dumps(fin)], 0.02), chat=True)

        async def go():
            return await asyncio.gather(events(self.pipe(up("AAA", 11), rid="a")),
                                        events(self.pipe(up("BBB", 22), rid="b")))
        a, b = run(go())
        self.assertEqual(a[0]["text"], "AAA")
        self.assertEqual(b[0]["text"], "BBB")
        rows = {r["request_id"]: r for r in self.store.recent(5)}
        self.assertEqual((rows["a"]["output_tokens"], rows["b"]["output_tokens"]), (11, 22))


# ---- CLI against a real local HTTP server (chunked NDJSON, like uvicorn/StreamingResponse) ------------
class _Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    scenario = "ok"
    store = None
    hub = None
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, s):
        data = s.encode("utf-8")
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        type(self).seen.append(body)
        sc = type(self).scenario
        if sc == "http500":
            self.send_response(500)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if sc == "cut":
            self._chunk(json.dumps({"type": "token", "text": "par"}) + "\n")
            self.close_connection = True
            return
        real = body["use_real_model"]

        def upstream():
            if sc == "mock_in_real":
                lines = [json.dumps({"type": "token", "text": f"token{i}"}) for i in range(1, 5)]
                lines.append(json.dumps({"type": "done", "mode": "SIMULATED", "output_tokens": 4}))
                return alines(lines)
            if sc == "ollama_down":
                return alines([json.dumps({"type": "error", "kind": "ollama_unreachable", "error": "Cannot connect to Ollama"})])
            if real:
                return parse_ollama_lines(alines([chat("Hel"), chat("lo \u0ba4\u0bae\u0bbf\u0bb4\u0bcd"), FINAL], 0.2), chat=True)
            return alines([json.dumps({"type": "token", "text": "token1"}),
                           json.dumps({"type": "done", "mode": "SIMULATED", "output_tokens": 1})])

        async def pump():
            async for line in instrument_stream(upstream(), store=type(self).store, hub=type(self).hub,
                                                request_id=body["request_id"], prompt=body["prompt"],
                                                model=MODEL, mode="REAL" if real else "SIMULATED",
                                                backend="model_a", policy="token_aware", profile="chat"):
                self._chunk(line)
        asyncio.run(pump())
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        _Gateway.store = MetricsStore(os.path.join(cls.tmp.name, "m.db"))
        _Gateway.hub = MetricsHub()
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        _Gateway.store._db.close()
        cls.tmp.cleanup()

    def go(self, scenario="ok", real=True, **kw):
        _Gateway.scenario = scenario
        buf = io.StringIO()
        res = cli.run_one("Explain scheduling", gateway=self.url, real=real, max_tokens=64, out=buf, **kw)
        return res, buf.getvalue()

    def test_payload_carries_real_mode_flag_and_token_limit(self):
        p = cli.build_payload("q", request_id="x", real=True, max_output_tokens=64)
        self.assertTrue(p["use_real_model"])
        self.assertEqual((p["max_output_tokens"], p["prompt"]), (64, "q"))
        self.assertIn("text", cli.build_payload("q", request_id="x", real=False, max_output_tokens=1, field="text"))
        self.assertFalse(cli.build_payload("q", request_id="x", real=False, max_output_tokens=1)["use_real_model"])

    def test_real_request_end_to_end_streams_and_matches_stored_row(self):
        res, shown = self.go("ok")
        self.assertEqual((res["status"], res["mode"]), ("SUCCESS", "REAL"))
        self.assertEqual(shown, "Hello \u0ba4\u0bae\u0bbf\u0bb4\u0bcd")        # streamed text, UTF-8 intact
        self.assertGreaterEqual(res["client_ttft_s"], 0.2)                     # TTFT measured at first token
        self.assertGreater(res["client_latency_s"], res["client_ttft_s"])
        self.assertEqual(_Gateway.seen[-1]["request_id"], res["request_id"])    # id preserved to the gateway
        self.assertTrue(_Gateway.seen[-1]["use_real_model"])
        d = res["done"]
        self.assertEqual((d["prompt_tokens"], d["output_tokens"]), (11, 40))
        self.assertAlmostEqual(d["tokens_per_second"], 5.0)
        row = next(r for r in _Gateway.store.recent(50) if r["request_id"] == res["request_id"])
        self.assertEqual((row["status"], row["mode"], row["output_tokens"]), ("SUCCESS", "REAL", 40))
        self.assertAlmostEqual(row["latency_s"], d["server_latency_s"], places=6)  # dashboard == terminal
        buf = io.StringIO()
        cli.print_summary(res, buf)
        text = buf.getvalue()
        for needle in ("Execution mode: REAL", "Output Tokens: 40", "Prompt Tokens: 11",
                       "Generation Throughput (model-only): 5.00 tokens/second", "Status: SUCCESS"):
            self.assertIn(needle, text)

    def test_mock_response_in_real_mode_is_never_reported_as_success(self):
        res, _ = self.go("mock_in_real")
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("invalid_real_response", res["error"])
        row = next(r for r in _Gateway.store.recent(50) if r["request_id"] == res["request_id"])
        self.assertEqual(row["status"], "FAILED")

    def test_simulated_mode_is_labelled(self):
        res, _ = self.go("ok", real=False)
        self.assertEqual((res["status"], res["mode"]), ("SUCCESS", "SIMULATED"))
        row = next(r for r in _Gateway.store.recent(50) if r["request_id"] == res["request_id"])
        self.assertEqual(row["mode"], "SIMULATED")

    def test_ollama_down_is_a_clear_failure(self):
        res, _ = self.go("ollama_down")
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("ollama_unreachable", res["error"])
        self.assertIsNone(res["done"].get("output_tokens"))

    def test_gateway_http_error(self):
        res, _ = self.go("http500")
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("HTTP 500", res["error"])

    def test_connection_cut_mid_stream(self):
        res, shown = self.go("cut")
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(shown, "par")
        self.assertTrue(res["error"])

    def test_gateway_unreachable(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        res = cli.run_one("hi", gateway=f"http://127.0.0.1:{port}", out=io.StringIO())
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("cannot reach the gateway", res["error"])

    def test_bench_summary_separates_failures(self):
        rows = [{"status": "SUCCESS", "latency_s": 2.0, "ttft_s": 0.4, "output_tokens": 40, "tokens_per_second": 5.0},
                {"status": "SUCCESS", "latency_s": 4.0, "ttft_s": None, "output_tokens": 60, "tokens_per_second": None},
                {"status": "FAILED", "latency_s": 50.0, "ttft_s": None, "output_tokens": None, "tokens_per_second": None}]
        a = aggregate(rows)
        self.assertEqual((a["total"], a["success"], a["failed"]), (3, 2, 1))
        self.assertAlmostEqual(a["avg_latency_s"], 3.0)
        self.assertEqual((a["n_ttft"], a["n_tokens_per_second"]), (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)

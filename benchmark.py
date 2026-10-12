import json
import time
import csv
import urllib.request
import urllib.error
from statistics import mean

URL = "http://127.0.0.1:9000/infer"
RUNS = 5

results = []

for i in range(1, RUNS + 1):
    payload = {
        "request_id": f"benchmark-{i:03d}",
        "prompt": "Explain token-aware scheduling in simple Tamil",
        "profile": "interactive",
        "max_output_tokens": 40,
        "policy": "token_aware",
        "simulate_failure": "none"
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()
    first_token_time = None
    token_events = 0
    generated_tokens = 0
    response_text = ""
    error = ""

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_events += 1
                    response_text += str(event.get("text", ""))

                elif event.get("type") == "done":
                    generated_tokens = int(
                        event.get("tokens_generated", token_events)
                    )

    except Exception as exc:
        error = str(exc)

    end = time.perf_counter()
    latency = end - start
    ttft = (
        first_token_time - start
        if first_token_time is not None
        else None
    )

    # This is an endpoint-level rate, not necessarily model-only speed.
    tps = generated_tokens / latency if latency > 0 else 0

    row = {
        "run": i,
        "latency_seconds": round(latency, 4),
        "ttft_seconds": round(ttft, 4) if ttft is not None else "",
        "output_tokens_reported": generated_tokens,
        "token_events": token_events,
        "tokens_per_second": round(tps, 3),
        "success": not bool(error) and bool(response_text),
        "error": error,
        "response": response_text
    }
    results.append(row)

    print(f"\nRun {i}")
    print("Latency (s):", row["latency_seconds"])
    print("TTFT (s):", row["ttft_seconds"])
    print("Output tokens reported:", generated_tokens)
    print("Token events:", token_events)
    print("Tokens/second:", row["tokens_per_second"])
    print("Success:", row["success"])
    if error:
        print("Error:", error)
    print("Response:", response_text[:300])

with open("benchmark_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

successful = [r for r in results if r["success"]]

print("\n=== SUMMARY ===")
print("Total requests:", len(results))
print("Successful requests:", len(successful))
print("Success rate (%):", round(100 * len(successful) / len(results), 2))

if successful:
    print("Average latency (s):",
          round(mean(r["latency_seconds"] for r in successful), 4))
    print("Average tokens/second:",
          round(mean(r["tokens_per_second"] for r in successful), 3))

print("\nSaved results to benchmark_results.csv")

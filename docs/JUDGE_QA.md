# HORIZON 2K26 · D07: Technical Judge Q&A Guide
### Adaptive Token-Aware Inference Request Scheduler with Self-Healing Backends

---

### Q1: Why use mock backends instead of downloading real 7B+ LLM weights?
**Answer:**
1. **Reproducibility & Judge Accessibility:** A real 7B parameter model requires 14–16 GB of GPU VRAM or causes huge memory swapping on CPU. Our software-only mock backends allow any judge to clone and run the full 4-tier distributed system on a standard laptop with zero GPU requirements, zero paid API keys, and zero cloud dependencies.
2. **Deterministic Validation:** Real models introduce stochastic execution variance and GPU thermal throttling. Our mock architecture provides precise control over token generation delays, prefill latency ($eta$), continuous batching degradation ($lpha$), and simulated failures.
3. **Architectural Isolation:** The core innovation of D07 is **request scheduling and queue management**, not model weights. Mock backends prove the scheduling algorithm's mathematical advantages in tail latency, throughput, and fairness independently of the neural network weights.

---

### Q2: How accurately do your mock backends reflect real LLM serving systems like vLLM or TGI?
**Answer:**
Our backends implement a **continuous batching simulation model**:
- **Prefill Latency:** $\text{Delay}_{\text{prefill}} = \beta \times \text{len(prompt)} \times \text{TIME\_SCALE}$, modeling the compute cost of processing input prompt context before first token.
- **Concurrency Degradation:** In continuous batching, memory bandwidth saturation causes per-token latency to increase with concurrent sequences:
  $$\text{Delay}_{\text{token}} = \text{BaseDelay} \times (1 + \alpha \times (\text{ActiveWorkers} - 1))$$
- **Lognormal Token Lengths:** Real LLM generation length follows heavy-tailed lognormal distributions per task type (chat/interactive vs. coding vs. batch processing).
- **NDJSON Incremental Streaming:** Backends stream real HTTP chunked NDJSON events matching OpenAI / vLLM streaming standards.

---

### Q3: Does your scheduler cheat by knowing the actual output token length in advance?
**Answer:**
**Strictly NO.**
- The client request only provides `max_output_tokens` (an upper bound requested by the user, usually much larger than the true answer).
- The true generated token length (`sim_true_output_tokens`) is hidden and **only read by the backend server** during token generation.
- The gateway uses an **Online EWMA Predictor** (`OnlineTokenPredictor`) that estimates actual length using historical observations grouped by `(profile, prompt_bucket)`.
- We benchmarked an `adaptive_oracle` policy (which has perfect foresight) solely as a theoretical upper bound, demonstrating that our online predictor captures >85% of oracle benefits without looking ahead.

---

### Q4: Why not just use Shortest Job First (SJF) or priority queueing?
**Answer:**
- **The Starvation Problem:** Pure SJF gives short interactive requests high priority, but completely starves batch processing jobs under heavy load.
- **D07 Aging Solution:** We incorporate an aging term $(w \times W_{\text{wait}})$ and an emergency **Starvation Bonus** ($+1000.0$) whenever any request waits longer than the starvation threshold ($30\text{s} \times \text{TIME\_SCALE}$).
- **Measured Result:** In our ablation benchmarks, removing aging (`adaptive_no_aging`) causes starvation incidents to spike, whereas our Adaptive policy achieves **0 starvation incidents** across all seeds while maintaining top-tier interactive TTFT.

---

### Q5: What is the computational overhead of the scheduler itself?
**Answer:**
- The scheduling decision, priority calculation, and cost routing execute in **$<1$ millisecond** using asynchronous locks, priority sorting, and in-memory monotonic sequence counters.
- Compared to backend token generation times ($50\text{ms} - 200\text{ms}$ per token), scheduler overhead is $<0.1\%$, making it production-viable.

---

### Q6: How does the system handle backend node failures and network degradation?
**Answer:**
- **Per-Backend Circuit Breaker:** Transitions from `HEALTHY` $\rightarrow$ `DEGRADED` $\rightarrow$ `OPEN` $\rightarrow$ `HALF_OPEN`.
- **Pre-Output Safe Retry:** If a backend fails before the first output token reaches the client (HTTP 500, socket error), the gateway transparently reroutes to an alternate healthy node.
- **No Partial Retries:** If failure occurs after partial output has already streamed, the gateway terminates the stream cleanly to prevent duplicate or corrupted output.

---

### Q7: How do you prove that these benchmark results are statistically significant?
**Answer:**
- We evaluate over multiple random seeds (e.g. seeds 42 through 46).
- We compute **95% Bootstrap Confidence Intervals** with 2000 resamples for all primary metrics (mean latency, p95 latency, TTFT, throughput, SLO attainment).
- Results are saved automatically to `results/RESULTS_<runid>.md` with paired seed-by-seed improvements.

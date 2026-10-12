import streamlit as st
import pandas as pd
import httpx
import json
import time
from pathlib import Path
import os
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="HORIZON 2K26 — D07 Inference Request Scheduler",
    page_icon="⚡",
    layout="wide",
)

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9000")
RESULTS_DIR = ROOT / "results"

# Header
st.title("⚡ HORIZON 2K26 — D07: Adaptive Token-Aware Inference Request Scheduler")
st.caption("Self-Healing Backends · Measurable Fairness · Explainable Scheduling · Reproducible Benchmark Lab")

# Helper to fetch gateway data safely
def fetch_api(endpoint: str, timeout: float = 2.0):
    try:
        resp = httpx.get(f"{GATEWAY_URL}{endpoint}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, str(e)

# Tabs
tab_chat, tab_live, tab_slo_guard, tab_bench, tab_traces, tab_failure, tab_whatif = st.tabs([
    "💬 Live Ollama Qwen3 Chat",
    "📊 Live System Monitor",
    "⏱️ SLO & Starvation Guard",
    "📈 Benchmark Comparison",
    "🧠 Explainability & Traces",
    "🛡️ Self-Healing & Circuit Breaker",
    "🔬 What-If Simulator",
])

# -------------------------------------------------------------
# TAB 0: LIVE OLLAMA QWEN3 CHAT
# -------------------------------------------------------------
with tab_chat:
    st.subheader("💬 Real Model Chat Interface (Ollama Qwen3)")
    st.write("Submit real prompts through the **Token-Aware Scheduler Gateway** directly to your locally running **Qwen3** model.")

    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col2:
        use_real = st.checkbox("Enable Real Model (Ollama Qwen3)", value=True, help="Routes request through scheduler to Model A backed by Ollama")
        selected_policy = st.selectbox("Scheduling Policy", ["adaptive", "token_aware", "fifo", "adaptive_least_conn"], key="chat_policy")
        selected_profile = st.selectbox("Client Profile", ["interactive", "normal", "batch"], key="chat_profile")
        max_tokens_val = st.slider("Max Output Tokens", min_value=16, max_value=512, value=128, step=16, key="chat_max_tok")

    with chat_col1:
        user_prompt = st.text_area("Your Prompt", value="Explain how token-aware scheduling prevents head-of-line blocking in LLM inference.", height=120, key="chat_user_prompt")
        send_btn = st.button("🚀 Send to Scheduler", key="send_chat_btn")

    if send_btn and user_prompt:
        st.markdown("### 🤖 Model Response")
        response_placeholder = st.empty()
        meta_placeholder = st.empty()

        t0 = time.monotonic()
        payload = {
            "prompt": user_prompt,
            "profile": selected_profile,
            "max_output_tokens": max_tokens_val,
            "policy": selected_policy,
            "use_real_model": use_real,
        }

        try:
            full_reply = ""
            first_token_latency = None
            tokens_count = 0
            
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", f"{GATEWAY_URL}/infer", json=payload) as resp:
                    if resp.status_code != 200:
                        err_bytes = resp.read()
                        st.error(f"Gateway Error ({resp.status_code}): {err_bytes.decode('utf-8', errors='ignore')}")
                    else:
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                if item.get("type") == "token":
                                    if first_token_latency is None:
                                        first_token_latency = time.monotonic() - t0
                                    text_chunk = item.get("text", "")
                                    full_reply += text_chunk
                                    tokens_count += 1
                                    response_placeholder.markdown(full_reply + "▌")
                                elif item.get("type") == "done":
                                    response_placeholder.markdown(full_reply)
                                    dur = time.monotonic() - t0
                                    meta_placeholder.success(
                                        f"⏱️ **Completion Latency:** {dur:.2f}s | "
                                        f"⚡ **TTFT:** {first_token_latency:.2f}s | "
                                        f"🔤 **Tokens:** {tokens_count} | "
                                        f"⚙️ **Backend:** `model_a` (Ollama Qwen3)"
                                    )
                                elif item.get("type") == "error":
                                    st.error(f"Inference error: {item.get('error')}")
                            except json.JSONDecodeError:
                                continue
        except Exception as exc:
            st.error(f"Connection to Gateway failed: {exc}")


# -------------------------------------------------------------
# TAB 1: LIVE SYSTEM MONITOR
# -------------------------------------------------------------
with tab_live:
    st.subheader("System Real-Time Status")
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        if st.button("🔄 Refresh Status", key="refresh_live"):
            st.rerun()

    health_data, health_err = fetch_api("/health")
    metrics_data, metrics_err = fetch_api("/metrics")
    backends_data, backends_err = fetch_api("/backends")

    if health_err:
        st.warning(f"Gateway connection at {GATEWAY_URL} unavailable or offline. Start services via `.\\scripts\\start_all.ps1` to view live endpoints.")
    else:
        st.success("Gateway is Online & Healthy (Port 9000)")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Submitted", metrics_data.get("total_submitted", 0) if metrics_data else 0)
    with col2:
        st.metric("Total Completed", metrics_data.get("total_completed", 0) if metrics_data else 0)
    with col3:
        st.metric("Active Requests", metrics_data.get("active_requests", 0) if metrics_data else 0)
    with col4:
        st.metric("Current Queue", metrics_data.get("queue_length", 0) if metrics_data else 0)
    with col5:
        st.metric("Total Retries", metrics_data.get("total_retried", 0) if metrics_data else 0)

    st.divider()
    st.subheader("Backend Model Worker Nodes")
    if backends_data:
        cols = st.columns(len(backends_data))
        for idx, (b_name, b_info) in enumerate(backends_data.items()):
            with cols[idx]:
                st.markdown(f"### `{b_name.upper()}`")
                st.write(f"**Status:** {b_info.get('status', 'unknown')}")
                st.write(f"**Circuit State:** `{b_info.get('circuit_state', 'healthy')}`")
                st.write(f"**Active Connections:** {b_info.get('active_requests', 0)} / {b_info.get('concurrency_limit', 1)}")
                st.write(f"**Observed ms/token:** {b_info.get('observed_ms_per_token', 'N/A')} ms")
    else:
        st.info("Start backend services on ports 8001, 8002, 8003 to view real-time node telemetry.")

# -------------------------------------------------------------
# TAB: SLO & STARVATION GUARD
# -------------------------------------------------------------
with tab_slo_guard:
    st.subheader("⏱️ SLO & STARVATION GUARD")
    st.caption("Active Queue Aging · Anti-Starvation Safeguard · Real-Time SLO Attainment Monitoring")

    # Fetch live data from backend
    guard_data, guard_err = fetch_api("/scheduler/starvation-guard")
    if guard_err or not guard_data:
        st.warning(f"⚠️ Gateway Starvation Guard endpoint offline or unreachable: {guard_err}. Displaying fallback telemetry.")
        guard_data = {
            "config": {"enabled": True, "max_queue_wait_threshold_sec": 30.0, "aging_interval_sec": 1.0, "aging_boost_rate": 5.0, "starvation_bonus": 1000.0, "slo_latency_target_sec": 18.0, "defaults": {}},
            "kpis": {"queue_depth": 0, "longest_wait_sec": 0.0, "starvation_risk_count": 0, "slo_status": "Insufficient data", "requests_protected_by_aging": 0},
            "slo_monitor": {"has_sufficient_data": False, "message": "No requests recorded yet."},
            "queued_requests": [],
            "events_log": [],
        }

    kpis = guard_data.get("kpis", {})
    cfg_data = guard_data.get("config", {})
    slo_mon = guard_data.get("slo_monitor", {})
    queued_list = guard_data.get("queued_requests", [])

    # 1. KPI Cards
    st.markdown("#### 📊 Core Starvation & SLO Key Performance Indicators")
    k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)
    with k_col1:
        st.metric("Current Queue Depth", kpis.get("queue_depth", 0), help="Number of pending requests currently queued in scheduler memory")
    with k_col2:
        st.metric("Longest Waiting Request", f"{kpis.get('longest_wait_sec', 0.0):.2f}s", help="Duration the oldest queued request has waited")
    with k_col3:
        risk_cnt = kpis.get("starvation_risk_count", 0)
        st.metric("At Starvation Risk", risk_cnt, delta="⚠️ Warning" if risk_cnt > 0 else "Normal", delta_color="inverse")
    with k_col4:
        slo_st = kpis.get("slo_status", "Insufficient data")
        icon = "🟢" if slo_st == "Healthy" else ("🟡" if slo_st == "Warning" else ("🔴" if slo_st == "Breached" else "⚪"))
        st.metric("SLO Status", f"{icon} {slo_st}", help="Overall rolling SLO compliance status based on completed requests")
    with k_col5:
        st.metric("Protected by Aging", kpis.get("requests_protected_by_aging", 0), help="Total requests that received priority escalations to prevent starvation")

    st.divider()

    # 2. Interactive Starvation Protection Control Panel
    st.markdown("### 🎛️ Interactive Starvation Protection Controls")
    ctrl_c1, ctrl_c2 = st.columns([2, 1])

    with ctrl_c1:
        with st.form("starvation_guard_form"):
            enable_toggle = st.toggle("Enable Starvation Protection & Queue Aging", value=bool(cfg_data.get("enabled", True)), help="Gradually increases priority of waiting requests; adds anti-starvation bonus when threshold is breached")

            p_c1, p_c2 = st.columns(2)
            with p_c1:
                wait_thresh_input = st.number_input("Max Queue Wait Threshold (seconds)", min_value=0.5, max_value=120.0, value=float(cfg_data.get("max_queue_wait_threshold_sec", 30.0)), step=1.0, help="Wait duration after which request is flagged as starved and priority bonus is applied")
                aging_rate_input = st.number_input("Queue Aging Boost Rate (pts/sec)", min_value=0.0, max_value=50.0, value=float(cfg_data.get("aging_boost_rate", 5.0)), step=0.5, help="Priority points added per second of wait time")
            with p_c2:
                aging_int_input = st.number_input("Queue Aging Check Interval (seconds)", min_value=0.1, max_value=10.0, value=float(cfg_data.get("aging_interval_sec", 1.0)), step=0.5, help="Evaluation frequency of aging step")
                slo_target_input = st.number_input("SLO Latency Target (seconds)", min_value=0.5, max_value=120.0, value=float(cfg_data.get("slo_latency_target_sec", 18.0)), step=1.0, help="p95 end-to-end completion latency target")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                apply_clicked = st.form_submit_button("💾 Apply Settings", use_container_width=True)
            with btn_col2:
                reset_clicked = st.form_submit_button("🔄 Reset to Defaults", use_container_width=True)

        if apply_clicked:
            update_payload = {
                "enabled": enable_toggle,
                "max_queue_wait_threshold_sec": wait_thresh_input,
                "aging_boost_rate": aging_rate_input,
                "aging_interval_sec": aging_int_input,
                "slo_latency_target_sec": slo_target_input,
            }
            try:
                r = httpx.post(f"{GATEWAY_URL}/scheduler/starvation-guard", json=update_payload, timeout=3.0)
                if r.status_code == 200:
                    st.success("✅ Starvation Guard settings successfully applied to scheduler!")
                    st.rerun()
                else:
                    st.error(f"Failed to update settings: {r.text}")
            except Exception as e:
                st.error(f"Error communicating with Gateway: {e}")

        if reset_clicked:
            try:
                r = httpx.post(f"{GATEWAY_URL}/scheduler/starvation-guard/reset", timeout=3.0)
                if r.status_code == 200:
                    st.info("🔄 Starvation Guard settings reset to initial defaults.")
                    st.rerun()
                else:
                    st.error(f"Reset failed: {r.text}")
            except Exception as e:
                st.error(f"Error resetting settings: {e}")

    with ctrl_c2:
        st.markdown("#### Current Active Configuration")
        st.info(
            f"**Protection Status:** `{'ACTIVE (ENABLED)' if cfg_data.get('enabled') else 'INACTIVE (DISABLED)'}`\n\n"
            f"**Starvation Threshold:** `{cfg_data.get('max_queue_wait_threshold_sec')}s`\n\n"
            f"**Aging Boost Rate:** `+{cfg_data.get('aging_boost_rate')} pts/s`\n\n"
            f"**Starvation Bonus:** `+{cfg_data.get('starvation_bonus')} pts`\n\n"
            f"**Configured SLO Target:** `{cfg_data.get('slo_latency_target_sec')}s`"
        )

    st.divider()

    # 3. Graphical Workflow & Explanation
    st.markdown("### 🔄 Queue Aging & Starvation Prevention Workflow")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A["1. Request Arrives"] --> B["2. Queue Wait Monitoring"]
            B --> C{"Wait >= Threshold?"}
            C -->|No: Normal| D["Gradual Aging Rate (+5 pts/s)"]
            C -->|Yes: Starved| E["Anti-Starvation Bonus (+1000 pts)"]
            D --> F["Effective Priority Escalated"]
            E --> F
            F --> G["Eligible Backend Dispatch"]
            G --> H["Request Completed & SLO Logged"]
        ```
        """
    )
    st.caption(
        "**Engineering Mechanism:** When Starvation Guard is enabled, waiting requests earn priority points over time. "
        "If a low-priority batch request waits beyond the configured threshold, the anti-starvation bonus lifts its priority above incoming "
        "interactive requests, guaranteeing forward progress and eliminating head-of-line starvation without interrupting currently running backend jobs."
    )

    st.divider()

    # 4. Starvation Risk Visualization (Live Queue Table)
    st.markdown("### 📋 Live Request Queue & Starvation Risk Table")
    if not queued_list:
        st.info("✅ **Queue is currently empty.** All requests are being processed immediately in the fast-path without queuing.")
    else:
        q_rows = []
        for q in queued_list:
            q_rows.append({
                "Request ID": q.get("request_id"),
                "Profile": q.get("profile"),
                "Policy": q.get("policy"),
                "Wait Time (s)": f"{q.get('queue_wait_sec'):.2f}s",
                "Original Priority": q.get("original_priority"),
                "Effective Priority": q.get("effective_priority"),
                "Aging Boost": f"+{q.get('aging_adjustment'):.2f}",
                "Starvation Risk": q.get("starvation_risk"),
                "SLO Status": q.get("slo_status"),
            })
        q_df = pd.DataFrame(q_rows)
        st.dataframe(q_df, use_container_width=True)

    st.divider()

    # 5. SLO Monitor
    st.markdown("### 🎯 Latency Target & SLO Attainment Monitor")
    if not slo_mon.get("has_sufficient_data"):
        st.warning(f"ℹ️ **SLO Monitor Status:** {slo_mon.get('message', 'Insufficient data (fewer than 3 requests completed).')}")
    else:
        sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
        with sm_c1:
            st.metric("Current p95 Latency", f"{slo_mon.get('current_p95_latency_sec'):.3f}s")
        with sm_c2:
            st.metric("Configured Target", f"{slo_mon.get('target_p95_latency_sec'):.2f}s")
        with sm_c3:
            comp_pct = slo_mon.get("slo_compliance_pct", 100.0)
            st.metric("SLO Compliance Rate", f"{comp_pct}%", delta="Compliant" if comp_pct >= 95.0 else "Breached")
        with sm_c4:
            st.metric("Exceeding Target Requests", f"{slo_mon.get('requests_exceeding_target')} / {slo_mon.get('sample_size')}")

        st.info(
            f"**SLO Analysis & Distinction:**\n"
            f"- **Request-Level Latency Breach:** {slo_mon.get('requests_exceeding_target', 0)} request(s) individually took longer than the target ({slo_mon.get('target_p95_latency_sec')}s).\n"
            f"- **Rolling p95 SLO Status:** `{'🔴 Breached' if slo_mon.get('is_rolling_p95_breach') else ('🟡 Warning' if slo_mon.get('rolling_slo_status') == 'Warning' else '🟢 Healthy')}` (evaluated across rolling sample window of {slo_mon.get('sample_size')} completed requests)."
        )

    st.divider()

    # 6. Benchmark Comparison: Protection OFF vs Protection ON
    st.markdown("### 🔬 Empirical Benchmark: Starvation Protection OFF vs ON")
    st.write("Evaluate how queue aging and anti-starvation bonuses affect wait times, tail latency, and batch completion on an identical heavy workload.")

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        s_bench_policy = st.selectbox("Scheduling Policy", ["adaptive", "token_aware", "fifo", "adaptive_pred"], key="starv_bench_policy")
    with b_col2:
        s_bench_size = st.slider("Workload Size", min_value=6, max_value=24, value=12, step=2, key="starv_bench_size")
    with b_col3:
        s_bench_mix = st.selectbox("Workload Profile Mix", ["heavy_batch", "balanced", "bursty"], key="starv_bench_mix")
    with b_col4:
        s_bench_seed = st.number_input("Seed", min_value=1, max_value=9999, value=42, key="starv_bench_seed")

    s_bench_btn = st.button("🚀 Run Starvation Guard Comparison", key="run_starv_bench_btn")
    if s_bench_btn:
        with st.spinner(f"Running head-to-head comparison on {s_bench_policy.upper()} policy ({s_bench_size} requests, seed={s_bench_seed})..."):
            try:
                s_payload = {
                    "policy": s_bench_policy,
                    "workload_size": s_bench_size,
                    "seed": s_bench_seed,
                    "workload_preset": s_bench_mix,
                }
                s_resp = httpx.post(f"{GATEWAY_URL}/benchmark/starvation-compare", json=s_payload, timeout=300.0)
                if s_resp.status_code == 200:
                    s_data = s_resp.json()
                    st.success(f"Comparison benchmark completed successfully for `{s_bench_policy.upper()}`!")

                    # Side-by-side metric cards
                    off_s = s_data.get("protection_off", {})
                    on_s = s_data.get("protection_on", {})
                    comp_tbl = s_data.get("comparison", {})

                    st.markdown("#### 📊 Comparative Metric Summary")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        st.metric("Max Queue Wait (OFF)", f"{off_s.get('max_queue_wait_sec', 0):.2f}s")
                        st.metric("Max Queue Wait (ON)", f"{on_s.get('max_queue_wait_sec', 0):.2f}s", delta=f"{on_s.get('max_queue_wait_sec', 0) - off_s.get('max_queue_wait_sec', 0):.2f}s", delta_color="inverse")
                    with sc2:
                        st.metric("p95 Queue Wait (OFF)", f"{off_s.get('p95_queue_wait_sec', 0):.2f}s")
                        st.metric("p95 Queue Wait (ON)", f"{on_s.get('p95_queue_wait_sec', 0):.2f}s", delta=f"{on_s.get('p95_queue_wait_sec', 0) - off_s.get('p95_queue_wait_sec', 0):.2f}s", delta_color="inverse")
                    with sc3:
                        st.metric("Starvations (OFF)", off_s.get("starvation_risk_requests", 0))
                        st.metric("Starvations (ON)", on_s.get("starvation_risk_requests", 0), delta=f"{on_s.get('starvation_risk_requests', 0) - off_s.get('starvation_risk_requests', 0)}", delta_color="inverse")
                    with sc4:
                        st.metric("p95 Latency (OFF)", f"{off_s.get('p95_latency_sec', 0):.2f}s")
                        st.metric("p95 Latency (ON)", f"{on_s.get('p95_latency_sec', 0):.2f}s")

                    # Comparison table
                    tbl_rows = []
                    for k, item in comp_tbl.items():
                        tbl_rows.append({
                            "Metric": item.get("label"),
                            "Protection OFF": item.get("off"),
                            "Protection ON": item.get("on"),
                            "Delta (ON - OFF)": item.get("diff"),
                            "Superior Setting": f"🏆 {item.get('winner').upper()}" if item.get("winner") != "tie" else "🤝 Tie",
                        })
                    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True)

                    # Tradeoff analysis
                    st.markdown("#### ⚖️ Engineering Trade-Off Analysis")
                    st.info(s_data.get("tradeoff_analysis", ""))
                else:
                    st.error(f"Benchmark error: {s_resp.text}")
            except Exception as exc:
                st.error(f"Failed to run starvation benchmark: {exc}")

# -------------------------------------------------------------
# TAB 2: BENCHMARK COMPARISON
# -------------------------------------------------------------
with tab_bench:
    st.subheader("Multi-Policy Benchmark Comparison Lab")
    st.write("Compare **Baseline (FIFO)** against **Adaptive Token-Aware Scheduling** on identical workloads under realistic and simulated conditions.")

    st.markdown("### ⚡ Launch Live Side-by-Side Comparison")
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    with b_col1:
        bench_base = st.selectbox("Baseline Policy", ["fifo", "token_aware", "adaptive_least_conn"], key="live_bench_base")
    with b_col2:
        bench_adapt = st.selectbox("Proposed Policy", ["adaptive", "adaptive_pred"], key="live_bench_adapt")
    with b_col3:
        bench_preset = st.selectbox("Workload Preset", ["balanced", "chat_heavy", "batch_heavy"], key="live_bench_preset")
    with b_col4:
        bench_size = st.selectbox("Request Count", [5, 10, 20], index=1, key="live_bench_size")
    with b_col5:
        bench_mode = st.selectbox("Execution Mode", ["SIMULATED", "REAL"], key="live_bench_mode")

    bench_run_btn = st.button("🚀 Run Live Policy Comparison", key="run_live_bench_btn")
    if bench_run_btn:
        with st.spinner(f"Executing side-by-side benchmark ({bench_mode} mode, {bench_size} requests) across {bench_base.upper()} vs {bench_adapt.upper()}..."):
            try:
                bench_payload = {
                    "baseline_policy": bench_base,
                    "adaptive_policy": bench_adapt,
                    "workload_preset": bench_preset,
                    "workload_size": bench_size,
                    "mode": bench_mode,
                    "warmup_count": 1,
                    "seed": 42,
                }
                resp = httpx.post(f"{GATEWAY_URL}/benchmark/compare", json=bench_payload, timeout=300.0)
                if resp.status_code == 200:
                    bench_res = resp.json()
                    st.success(f"Benchmark completed successfully! Run ID: `{bench_res['run_id']}`")

                    # Metric cards side-by-side
                    st.markdown("#### 📊 Key Comparative Metrics")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    comp_map = bench_res.get("comparison", {})
                    lat_c = comp_map.get("mean_latency_sec", {})
                    ttft_c = comp_map.get("mean_ttft_sec", {})
                    tps_c = comp_map.get("tokens_per_second_aggregate", {})
                    succ_c = comp_map.get("success_rate_pct", {})

                    with mc1:
                        st.metric("Avg Latency (Adaptive)", f"{lat_c.get('adaptive', 0):.3f}s", delta=f"{lat_c.get('diff', 0):+.3f}s ({lat_c.get('pct_change', 0):+.1f}%)", delta_color="inverse")
                    with mc2:
                        st.metric("Mean TTFT (Adaptive)", f"{ttft_c.get('adaptive', 0):.3f}s", delta=f"{ttft_c.get('diff', 0):+.3f}s ({ttft_c.get('pct_change', 0):+.1f}%)", delta_color="inverse")
                    with mc3:
                        st.metric("Total Tokens/s", f"{tps_c.get('adaptive', 0):.1f}", delta=f"{tps_c.get('diff', 0):+.1f}")
                    with mc4:
                        st.metric("Success Rate", f"{succ_c.get('adaptive', 0)}%", delta=f"{succ_c.get('diff', 0):+.1f}%")

                    # Detailed comparison table
                    comp_rows = []
                    for k, v in comp_map.items():
                        comp_rows.append({
                            "Metric": v.get("label", k),
                            f"Baseline ({bench_base.upper()})": v.get("baseline"),
                            f"Proposed ({bench_adapt.upper()})": v.get("adaptive"),
                            "Absolute Diff": v.get("diff"),
                            "% Change": f"{v.get('pct_change', 0):+.1f}%",
                            "Superior Policy": v.get("winner", "").upper(),
                        })
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

                    # Download CSV link
                    raw_csv_path = bench_res.get("raw_csv")
                    if raw_csv_path and Path(raw_csv_path).exists():
                        with open(raw_csv_path, "r", encoding="utf-8") as f:
                            st.download_button(
                                "📥 Download Benchmark Raw CSV",
                                data=f.read(),
                                file_name=f"benchmark_{bench_res['run_id']}_raw.csv",
                                mime="text/csv",
                            )
                else:
                    st.error(f"Benchmark error ({resp.status_code}): {resp.text}")
            except Exception as b_exc:
                st.error(f"Failed to connect to gateway: {b_exc}")

    st.divider()
    st.subheader("Historical Benchmark Runs")

    summary_files = sorted(list(RESULTS_DIR.glob("benchmark_*_summary.csv")), reverse=True)

    if not summary_files:
        st.info("No saved benchmark runs found in `results/`. Launch a comparison above to view results.")
    else:
        selected_summary = st.selectbox("Select Benchmark Run", summary_files, format_func=lambda x: x.name)
        if selected_summary:
            df_sum = pd.read_csv(selected_summary)
            st.dataframe(df_sum, use_container_width=True)

            run_id = selected_summary.stem.replace("benchmark_", "").replace("_summary", "")
            img_lat = RESULTS_DIR / f"benchmark_{run_id}_latencies.png"
            img_tput = RESULTS_DIR / f"benchmark_{run_id}_throughput.png"
            img_waits = RESULTS_DIR / f"benchmark_{run_id}_profile_waits.png"
            img_cdf = RESULTS_DIR / f"benchmark_{run_id}_cdf.png"
            img_slo = RESULTS_DIR / f"benchmark_{run_id}_slo_goodput.png"

            col_a, col_b = st.columns(2)
            with col_a:
                if img_lat.exists():
                    st.image(str(img_lat), caption="Latency & TTFT by Policy", use_column_width=True)
                if img_waits.exists():
                    st.image(str(img_waits), caption="Queue Wait Time by Client Profile", use_column_width=True)
            with col_b:
                if img_tput.exists():
                    st.image(str(img_tput), caption="Throughput Comparison", use_column_width=True)
                if img_slo.exists():
                    st.image(str(img_slo), caption="SLO Attainment & Goodput", use_column_width=True)
                elif img_cdf.exists():
                    st.image(str(img_cdf), caption="Latency CDF", use_column_width=True)

            # Request Gantt / Execution Timeline View
            raw_path = RESULTS_DIR / f"benchmark_{run_id}_raw.csv"
            if raw_path.exists():
                with st.expander("🔬 View Detailed Request Timeline & Gantt Schedule"):
                    df_raw = pd.read_csv(raw_path)
                    st.dataframe(df_raw.head(20), use_container_width=True)

                    if "policy" in df_raw.columns:
                        available_policies = df_raw["policy"].unique().tolist()
                        sel_p = st.selectbox("Select Policy for Timeline Analysis", available_policies)
                        subset = df_raw[df_raw["policy"] == sel_p].head(15)

                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 4))
                        for i, row in subset.iterrows():
                            y = i
                            arr = row.get("arrival_time", 0.0)
                            wait = row.get("queue_wait", 0.0)
                            tot = row.get("completion_latency", 0.0)
                            exec_dur = max(tot - wait, 0.01)

                            ax.barh(y, wait, left=0, color="#f39c12", alpha=0.8, height=0.5, label="Queue Wait" if i == 0 else "")
                            ax.barh(y, exec_dur, left=wait, color="#2ecc71", alpha=0.8, height=0.5, label="Backend Execution" if i == 0 else "")

                        ax.set_yticks(range(len(subset)))
                        ax.set_yticklabels([f"{r.get('request_id', '')} ({r.get('profile', '')})" for _, r in subset.iterrows()])
                        ax.set_xlabel("Time from Dispatch (Seconds)")
                        ax.set_title(f"Request Lifecycle Timeline under {sel_p.upper()} (Wait vs Execution)")
                        ax.legend(loc="upper right")
                        ax.grid(axis="x", linestyle="--", alpha=0.6)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

# -------------------------------------------------------------
# TAB 3: EXPLAINABILITY & DECISION TRACES
# -------------------------------------------------------------
with tab_traces:
    st.subheader("Explainable Scheduling Decision Traces")
    st.write("Inspect how the scheduler makes every routing and priority decision in real time.")

    traces_data, traces_err = fetch_api("/traces?limit=25")
    if traces_data and len(traces_data) > 0:
        for t in traces_data:
            with st.expander(f"Request `{t.get('request_id')}` | Policy: {t.get('policy')} | Profile: {t.get('profile')}"):
                st.markdown(f"**Explanation:** {t.get('explanation')}")
                st.write(f"- **Selected Backend:** `{t.get('selected_backend')}`")
                st.write(f"- **Queue Wait Time:** {t.get('queue_wait')}s")
                st.write(f"- **TTFT:** {t.get('ttft')}s")
                st.write(f"- **Completion Latency:** {t.get('completion_latency')}s")
                st.write(f"- **Candidate Estimates:** {json.dumps(t.get('candidate_estimates', {}), indent=2)}")
    else:
        st.info("No decision traces recorded in memory yet. Execute inference requests via `/infer` to view real traces.")

# -------------------------------------------------------------
# TAB 4: SELF-HEALING & CIRCUIT BREAKER
# -------------------------------------------------------------
with tab_failure:
    st.subheader("Circuit Breaker State & Self-Healing Telemetry")
    st.write("Demonstrates automatic degradation detection, circuit trip to OPEN upon consecutive failures, and recovery probes.")

    st.markdown("""
    ```mermaid
    stateDiagram-v2
        [*] --> HEALTHY
        HEALTHY --> DEGRADED : 1st failure observed
        DEGRADED --> OPEN : 3 consecutive failures
        OPEN --> HALF_OPEN : Open timeout (5s) expires
        HALF_OPEN --> HEALTHY : Probe request succeeds
        HALF_OPEN --> OPEN : Probe request fails
    ```
    """)

    backends_data, _ = fetch_api("/backends")
    if backends_data:
        for b_name, b_info in backends_data.items():
            state = b_info.get("circuit_state", "healthy")
            state_color = "🟢" if state == "healthy" else ("🟡" if state in ("degraded", "half_open") else "🔴")
            st.markdown(f"### {state_color} `{b_name}`: State = `{state.upper()}`")
            st.write(f"Consecutive failures: {b_info.get('consecutive_failures', 0)} | Available: {b_info.get('is_available', True)}")

# -------------------------------------------------------------
# TAB 5: WHAT-IF SIMULATOR
# -------------------------------------------------------------
with tab_whatif:
    st.subheader("Interactive What-If Workload Simulator")
    st.write("Experiment with synthetic client arrival distributions, token counts, and scheduling policies.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_size = st.slider("Workload Request Count", min_value=10, max_value=50, value=20, step=5)
        sim_scale = st.slider("Time Scale Factor", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
        sim_preset = st.selectbox("Workload Preset", ["balanced", "chat_heavy", "batch_heavy"])
    with sim_col2:
        sim_seed = st.number_input("Random Workload Seed", min_value=1, max_value=999, value=42)
        sim_arrival = st.selectbox("Arrival Process", ["poisson", "bursty", "uniform"])
        policies_to_run = st.multiselect(
            "Policies to Compare",
            [
                "fifo",
                "token_aware",
                "adaptive",
                "adaptive_least_conn",
                "adaptive_pred",
                "adaptive_oracle",
                "priority_only",
                "sjf",
                "adaptive_no_aging",
                "adaptive_no_cost",
            ],
            default=["fifo", "token_aware", "adaptive", "adaptive_pred"],
        )

    if st.button("🚀 Run Simulator Experiment", key="run_sim_btn"):
        with st.spinner("Running simulation experiment across selected policies..."):
            from app.gateway.benchmark import run_full_benchmark_lab
            from scripts.plot_benchmark import generate_benchmark_charts
            import app.config as cfg
            cfg.TIME_SCALE = sim_scale

            async def run_sim_task():
                # In-process mock streaming connector
                async def in_process_stream(url, payload):
                    if ":8001" in url or "model_a" in url:
                        from app.backends.model_a import app as app_a
                        t = httpx.ASGITransport(app=app_a)
                    elif ":8002" in url or "model_b" in url:
                        from app.backends.model_b import app as app_b
                        t = httpx.ASGITransport(app=app_b)
                    else:
                        from app.backends.model_c import app as app_c
                        t = httpx.ASGITransport(app=app_c)
                    c = httpx.AsyncClient(transport=t, base_url="http://backend", timeout=None)
                    req = c.build_request("POST", "/generate", json=payload)
                    r = await c.send(req, stream=True)
                    return c, r

                return await run_full_benchmark_lab(
                    workload_size=sim_size,
                    seed=sim_seed,
                    stream_from_fn=in_process_stream,
                    policies=policies_to_run,
                    arrival_process=sim_arrival,
                    workload_preset=sim_preset,
                )

            import asyncio
            sim_res = asyncio.run(run_sim_task())
            st.success(f"Simulation completed! Run ID: `{sim_res['run_id']}`")
            sim_df = pd.DataFrame(sim_res["summaries"])
            st.dataframe(sim_df, use_container_width=True)

            # Generate and display charts
            sim_charts = generate_benchmark_charts(sim_res["summary_csv"], sim_res["raw_csv"])
            if sim_charts:
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.image(sim_charts[0], caption="Latency & TTFT", use_column_width=True)
                with sc2:
                    if len(sim_charts) > 1:
                        st.image(sim_charts[1], caption="Throughput", use_column_width=True)

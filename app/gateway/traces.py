import time
from typing import Dict, Any, List, Optional
from collections import deque

class DecisionTraceManager:
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._traces: deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._by_id: Dict[str, Dict[str, Any]] = {}

    def reset(self):
        self._traces.clear()
        self._by_id.clear()

    def record_trace(self, trace: Dict[str, Any]):
        # Ensure timestamp
        if "timestamp" not in trace:
            trace["timestamp"] = time.time()
        self._traces.append(trace)
        req_id = trace.get("request_id")
        if req_id:
            self._by_id[req_id] = trace
            # Keep lookup map bounded
            if len(self._by_id) > self.capacity * 2:
                # Evict oldest
                oldest_keys = list(self._by_id.keys())[:len(self._by_id) - self.capacity]
                for k in oldest_keys:
                    self._by_id.pop(k, None)

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        traces = list(self._traces)
        traces.reverse()
        return traces[:limit]

    def get_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(request_id)

    @staticmethod
    def generate_explanation(
        policy: str,
        profile: str,
        selected_backend: Optional[str],
        queue_wait: float,
        score_breakdown: Dict[str, Any],
        candidate_estimates: Dict[str, Any],
        rejected_reasons: Dict[str, str],
        success: bool = True,
        error_detail: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        fallback_occurred: bool = False,
    ) -> str:
        if not success:
            reasons_str = "; ".join(f"{b}: {r}" for b, r in rejected_reasons.items()) or "all backends unavailable"
            return f"{policy.upper()} dispatch failed for {profile} request after {queue_wait:.2f}s queue wait: {error_detail or reasons_str}."

        fallback_txt = " [Fallback: primary backend failed; successfully rerouted to secondary backend]" if fallback_occurred else ""

        if policy == "fifo":
            act = candidate_estimates.get(selected_backend, {}).get("active", 0)
            return (
                f"FIFO policy dispatched {profile} request to {selected_backend} (arrived earliest in queue, "
                f"waited {queue_wait:.2f}s, backend active: {act}).{fallback_txt}"
            )

        if policy == "token_aware":
            score = score_breakdown.get("total_score", 0.0)
            act = candidate_estimates.get(selected_backend, {}).get("active", 0)
            return (
                f"Token-Aware policy selected {profile} request (score: {score:.1f}, "
                f"max output: {max_output_tokens or 'N/A'} tokens) after {queue_wait:.2f}s wait; "
                f"routed to {selected_backend} ({act} in-flight requests).{fallback_txt}"
            )

        # Adaptive and related cost-aware variants
        est = candidate_estimates.get(selected_backend, {})
        est_sec = est.get("estimated_completion_sec", 0.0)
        ms_tok = est.get("ms_per_token", 0.0)
        act = est.get("active", 0)
        tps = est.get("observed_tps") or (round(1000.0 / ms_tok, 1) if ms_tok > 0 else "N/A")

        rejections = [f"{b} ({r})" for b, r in rejected_reasons.items()]
        rejection_txt = f" Excluded unhealthy/busy: {', '.join(rejections)}." if rejections else ""

        others = []
        for b, v in candidate_estimates.items():
            if b != selected_backend and isinstance(v, dict):
                c_sec = v.get("estimated_completion_sec")
                if c_sec is not None:
                    others.append(f"{b} ({c_sec:.2f}s)")
        comp_txt = f" Outperformed alternatives: {', '.join(others)}." if others else ""
        score_val = score_breakdown.get("total_score", 0.0)
        pol_label = "Adaptive" if policy == "adaptive" else ("Adaptive Token-Aware" if "token_aware" in policy else policy.capitalize())
        return (
            f"{pol_label} policy scheduled {profile} request (score: {score_val:.1f}, "
            f"est input: {prompt_tokens or 'N/A'} tok, max output: {max_output_tokens or 'N/A'} tok) "
            f"after {queue_wait:.2f}s wait. Selected {selected_backend} because its measured service time "
            f"({ms_tok:.0f}ms/tok, ~{tps} tok/s) and queue state ({act} in-flight) produced lowest expected "
            f"completion cost ({est_sec:.2f}s).{comp_txt}{rejection_txt}{fallback_txt}"
        )

# Global singleton
trace_manager = DecisionTraceManager(capacity=200)

"""SLO-Aware Starvation Prevention Guard.

Provides an additional protection layer on top of existing policies (FIFO, Token-Aware,
Adaptive, Adaptive Pred) to dynamically prevent queue starvation and monitor SLO attainment.
"""
from typing import Dict, Any, List, Optional, Tuple
import threading
import time
import numpy as np

from app.config import (
    TIME_SCALE,
    STARVATION_WAIT_THRESHOLD,
    STARVATION_BONUS,
    SLO_TARGETS,
    PROFILE_SCORES,
    POLICY_FIFO,
    POLICY_TOKEN_AWARE,
    POLICY_ADAPTIVE,
    POLICY_ADAPTIVE_PRED,
    POLICY_PRIORITY_ONLY,
    POLICY_SJF,
)


class StarvationGuard:
    """Thread-safe Starvation Prevention and SLO Monitor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.default_enabled: bool = True
        self.default_max_queue_wait_threshold_sec: float = float(STARVATION_WAIT_THRESHOLD)
        self.default_aging_interval_sec: float = 1.0 * TIME_SCALE
        self.default_aging_boost_rate: float = 5.0
        self.default_starvation_bonus: float = float(STARVATION_BONUS)
        self.default_slo_latency_target_sec: float = 18.0 * TIME_SCALE
        self.warning_threshold_ratio: float = 0.6  # 60% of threshold triggers warning

        self.reset()

    def reset(self) -> None:
        """Reset configuration and state to defaults."""
        with self._lock:
            self.enabled: bool = self.default_enabled
            self.max_queue_wait_threshold_sec: float = self.default_max_queue_wait_threshold_sec
            self.aging_interval_sec: float = self.default_aging_interval_sec
            self.aging_boost_rate: float = self.default_aging_boost_rate
            self.starvation_bonus: float = self.default_starvation_bonus
            self.slo_latency_target_sec: float = self.default_slo_latency_target_sec
            self.requests_protected_count: int = 0
            self.starvation_events_log: List[Dict[str, Any]] = []

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration and defaults."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "max_queue_wait_threshold_sec": round(self.max_queue_wait_threshold_sec, 2),
                "aging_interval_sec": round(self.aging_interval_sec, 2),
                "aging_boost_rate": round(self.aging_boost_rate, 2),
                "starvation_bonus": round(self.starvation_bonus, 2),
                "slo_latency_target_sec": round(self.slo_latency_target_sec, 2),
                "defaults": {
                    "enabled": self.default_enabled,
                    "max_queue_wait_threshold_sec": round(self.default_max_queue_wait_threshold_sec, 2),
                    "aging_interval_sec": round(self.default_aging_interval_sec, 2),
                    "aging_boost_rate": round(self.default_aging_boost_rate, 2),
                    "starvation_bonus": round(self.default_starvation_bonus, 2),
                    "slo_latency_target_sec": round(self.default_slo_latency_target_sec, 2),
                },
            }

    def update_config(self, settings: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate and apply runtime configuration updates."""
        with self._lock:
            # 1. enabled
            if "enabled" in settings:
                self.enabled = bool(settings["enabled"])

            # 2. max_queue_wait_threshold_sec
            if "max_queue_wait_threshold_sec" in settings:
                try:
                    val = float(settings["max_queue_wait_threshold_sec"])
                    if val <= 0.0 or val > 300.0:
                        return False, "max_queue_wait_threshold_sec must be between 0.1 and 300.0 seconds"
                    self.max_queue_wait_threshold_sec = val
                except (ValueError, TypeError):
                    return False, "Invalid number for max_queue_wait_threshold_sec"

            # 3. aging_interval_sec
            if "aging_interval_sec" in settings:
                try:
                    val = float(settings["aging_interval_sec"])
                    if val <= 0.0 or val > 60.0:
                        return False, "aging_interval_sec must be between 0.05 and 60.0 seconds"
                    self.aging_interval_sec = val
                except (ValueError, TypeError):
                    return False, "Invalid number for aging_interval_sec"

            # 4. aging_boost_rate
            if "aging_boost_rate" in settings:
                try:
                    val = float(settings["aging_boost_rate"])
                    if val < 0.0 or val > 100.0:
                        return False, "aging_boost_rate must be between 0.0 and 100.0"
                    self.aging_boost_rate = val
                except (ValueError, TypeError):
                    return False, "Invalid number for aging_boost_rate"

            # 5. slo_latency_target_sec
            if "slo_latency_target_sec" in settings:
                try:
                    val = float(settings["slo_latency_target_sec"])
                    if val <= 0.0 or val > 300.0:
                        return False, "slo_latency_target_sec must be between 0.1 and 300.0 seconds"
                    self.slo_latency_target_sec = val
                except (ValueError, TypeError):
                    return False, "Invalid number for slo_latency_target_sec"

            return True, "Settings updated successfully"

    def record_protection_event(self, request_id: str, profile: str, wait_sec: float, boost: float) -> None:
        """Record that a request received an aging-based priority adjustment."""
        with self._lock:
            self.requests_protected_count += 1
            entry = {
                "timestamp": time.time(),
                "request_id": request_id,
                "profile": profile,
                "wait_sec": round(wait_sec, 3),
                "priority_boost": round(boost, 2),
            }
            self.starvation_events_log.append(entry)
            if len(self.starvation_events_log) > 200:
                self.starvation_events_log.pop(0)

    def calculate_starvation_risk(self, wait_sec: float) -> Tuple[str, str]:
        """Classify starvation risk. Returns: (risk_label, color_code)."""
        threshold = self.max_queue_wait_threshold_sec
        warn_point = threshold * self.warning_threshold_ratio
        if wait_sec >= threshold:
            return "Threshold Exceeded (Starved)", "red"
        if wait_sec >= warn_point:
            return "Approaching Threshold (Warning)", "yellow"
        return "Healthy", "green"

    def calculate_request_slo_status(self, wait_sec: float, profile: str = "normal", max_tokens: int = 5) -> Tuple[str, str]:
        """Classify SLO status for a queued request. Returns: (status_label, color_code)."""
        target = self.slo_latency_target_sec
        est_service_sec = (max_tokens * 100.0 * TIME_SCALE) / 1000.0
        projected_lat = wait_sec + est_service_sec

        if projected_lat > target:
            return "Breached", "red"
        if projected_lat > target * 0.75:
            return "Warning", "yellow"
        return "Healthy", "green"

    def inspect_queue_items(self, queue_items: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
        """Compute detailed starvation risk, original vs effective priority, and SLO metrics for queue."""
        result = []
        for req in queue_items:
            req_id = req.get("request_id", "unknown")
            profile = req.get("profile", "normal")
            policy = req.get("policy", POLICY_FIFO)
            enqueued_at = req.get("enqueued_at", now)
            wait_sec = max(0.0, now - enqueued_at)
            max_tokens = req.get("max_output_tokens", 5)

            # 1. Original Priority (baseline score before waiting/aging/starvation bonus)
            profile_base = float(PROFILE_SCORES.get(profile, 50))
            if policy in (POLICY_FIFO,):
                # For FIFO, arrival sequence determines baseline order
                orig_priority = round(float(req.get("arrival_seq", 0)), 1)
            elif policy in (POLICY_PRIORITY_ONLY,):
                orig_priority = profile_base
            elif policy in (POLICY_SJF,):
                orig_priority = -float(max_tokens)
            elif policy in (POLICY_TOKEN_AWARE,):
                orig_priority = round(profile_base - (max_tokens * 0.2), 2)
            else:
                # Adaptive & Adaptive Pred
                tokens = req.get("predicted_tokens", max_tokens) if policy == POLICY_ADAPTIVE_PRED else max_tokens
                est_cost = (tokens * 100.0 * TIME_SCALE) / 1000.0
                orig_priority = round(profile_base - (est_cost * 2.0), 2)

            # 2. Effective Priority (with aging adjustment if enabled)
            aging_adj = 0.0
            is_protected = False
            starvation_bonus_active = False

            if self.enabled:
                aging_adj += wait_sec * self.aging_boost_rate
                if wait_sec >= self.max_queue_wait_threshold_sec:
                    aging_adj += self.starvation_bonus
                    starvation_bonus_active = True
                if aging_adj > 0.05:
                    is_protected = True
                effective_priority = round(orig_priority + aging_adj, 2)
            else:
                effective_priority = orig_priority

            risk_label, risk_color = self.calculate_starvation_risk(wait_sec)
            slo_label, slo_color = self.calculate_request_slo_status(wait_sec, profile, max_tokens)

            result.append({
                "request_id": req_id,
                "profile": profile,
                "policy": policy,
                "max_output_tokens": max_tokens,
                "arrival_seq": req.get("arrival_seq", 0),
                "queue_wait_sec": round(wait_sec, 3),
                "original_priority": orig_priority,
                "effective_priority": effective_priority,
                "aging_adjustment": round(aging_adj, 2),
                "is_protected_by_aging": is_protected,
                "starvation_bonus_active": starvation_bonus_active,
                "starvation_risk": risk_label,
                "starvation_risk_color": risk_color,
                "slo_status": slo_label,
                "slo_status_color": slo_color,
            })

        return result

    def compute_kpis(
        self,
        queue_items: List[Dict[str, Any]],
        now: float,
        metrics: Dict[str, Any],
        recent_latencies: List[float],
    ) -> Dict[str, Any]:
        """Compute the 5 required KPI cards based on live scheduler state."""
        queue_depth = len(queue_items)

        if queue_items:
            waits = [max(0.0, now - r.get("enqueued_at", now)) for r in queue_items]
            longest_wait_sec = round(max(waits), 3)
            threshold = self.max_queue_wait_threshold_sec
            warn_point = threshold * self.warning_threshold_ratio
            starvation_risk_count = sum(1 for w in waits if w >= warn_point)
        else:
            longest_wait_sec = 0.0
            starvation_risk_count = 0

        # Protected by aging requests count
        protected_count = self.requests_protected_count + metrics.get("requests_protected_by_aging", 0)

        # Rolling SLO status based on completed latencies
        if len(recent_latencies) < 3:
            slo_status = "Insufficient data"
            slo_status_color = "gray"
        else:
            p95 = float(np.percentile(recent_latencies, 95))
            target = self.slo_latency_target_sec
            if p95 <= target:
                slo_status = "Healthy"
                slo_status_color = "green"
            elif p95 <= target * 1.25:
                slo_status = "Warning"
                slo_status_color = "yellow"
            else:
                slo_status = "Breached"
                slo_status_color = "red"

        return {
            "queue_depth": queue_depth,
            "longest_wait_sec": longest_wait_sec,
            "starvation_risk_count": starvation_risk_count,
            "slo_status": slo_status,
            "slo_status_color": slo_status_color,
            "requests_protected_by_aging": protected_count,
            "is_protection_enabled": self.enabled,
        }

    def compute_slo_monitor(
        self,
        recent_latencies: List[float],
        rolling_window: int = 50,
    ) -> Dict[str, Any]:
        """Compute SLO monitor tracking completed requests vs configured target."""
        target = round(self.slo_latency_target_sec, 2)
        sample = recent_latencies[-rolling_window:] if recent_latencies else []
        n_samples = len(sample)

        if n_samples < 3:
            return {
                "has_sufficient_data": False,
                "sample_size": n_samples,
                "min_sample_size": 3,
                "current_p95_latency_sec": None,
                "target_p95_latency_sec": target,
                "slo_compliance_pct": None,
                "requests_exceeding_target": 0,
                "rolling_slo_status": "Insufficient data",
                "request_level_breaches": 0,
                "message": f"Insufficient data ({n_samples}/3 requests completed). Submit requests to populate SLO monitor.",
            }

        p95 = round(float(np.percentile(sample, 95)), 3)
        mean_lat = round(float(np.mean(sample)), 3)
        exceeding = sum(1 for lat in sample if lat > target)
        compliance_pct = round(100.0 * (1.0 - (exceeding / n_samples)), 1)

        # Distinguish request-level breach from rolling p95 breach
        is_p95_breach = p95 > target
        is_p95_warning = not is_p95_breach and (p95 > target * 0.8)

        if is_p95_breach:
            rolling_status = "Breached"
        elif is_p95_warning:
            rolling_status = "Warning"
        else:
            rolling_status = "Healthy"

        return {
            "has_sufficient_data": True,
            "sample_size": n_samples,
            "current_p95_latency_sec": p95,
            "mean_latency_sec": mean_lat,
            "target_p95_latency_sec": target,
            "slo_compliance_pct": compliance_pct,
            "requests_exceeding_target": exceeding,
            "rolling_slo_status": rolling_status,
            "is_rolling_p95_breach": is_p95_breach,
            "is_request_level_breach_present": exceeding > 0,
        }


# Global singleton
starvation_guard = StarvationGuard()

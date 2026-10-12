import time
from typing import Dict, Any, List, Optional
from ..config import (
    BACKEND_URLS,
    CB_FAILURE_THRESHOLD,
    CB_OPEN_DURATION,
    CB_HALF_OPEN_PROBES,
    TOKEN_DELAY_MS,
    TIME_SCALE,
)

# Circuit Breaker states
STATE_HEALTHY = "healthy"
STATE_DEGRADED = "degraded"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

class BackendHealthManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self._states: Dict[str, str] = {name: STATE_HEALTHY for name in BACKEND_URLS}
        self._consecutive_failures: Dict[str, int] = {name: 0 for name in BACKEND_URLS}
        self._last_state_change: Dict[str, float] = {name: time.monotonic() for name in BACKEND_URLS}
        self._half_open_probes: Dict[str, int] = {name: 0 for name in BACKEND_URLS}
        # EWMA observed ms per token
        self._observed_ms_per_token: Dict[str, float] = {
            name: float(TOKEN_DELAY_MS.get(name, 100)) * TIME_SCALE for name in BACKEND_URLS
        }
        self._events: List[Dict[str, Any]] = []

    def _log_event(self, backend: str, from_state: str, to_state: str, reason: str):
        event = {
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "backend": backend,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
        }
        self._events.append(event)
        if len(self._events) > 500:
            self._events.pop(0)

    def get_state(self, backend: str) -> str:
        current = self._states.get(backend, STATE_HEALTHY)
        now = time.monotonic()
        # If open and open duration elapsed, transition to half-open
        if current == STATE_OPEN:
            elapsed = now - self._last_state_change[backend]
            if elapsed >= CB_OPEN_DURATION:
                self._states[backend] = STATE_HALF_OPEN
                self._last_state_change[backend] = now
                self._half_open_probes[backend] = 0
                self._log_event(backend, STATE_OPEN, STATE_HALF_OPEN, f"Open duration {CB_OPEN_DURATION:.1f}s expired; probing")
                return STATE_HALF_OPEN
        return self._states.get(backend, STATE_HEALTHY)

    def is_available(self, backend: str) -> bool:
        state = self.get_state(backend)
        if state in (STATE_HEALTHY, STATE_DEGRADED):
            return True
        if state == STATE_HALF_OPEN:
            # Allow limited probe requests
            return self._half_open_probes[backend] < CB_HALF_OPEN_PROBES
        return False

    def record_success(self, backend: str, latency_sec: float, tokens: int):
        now = time.monotonic()
        prev_state = self.get_state(backend)
        self._consecutive_failures[backend] = 0

        # Update observed ms per token (EWMA alpha=0.3)
        if tokens > 0 and latency_sec > 0:
            sample_ms_per_tok = (latency_sec * 1000.0) / tokens
            current_est = self._observed_ms_per_token[backend]
            self._observed_ms_per_token[backend] = 0.7 * current_est + 0.3 * sample_ms_per_tok

        if prev_state in (STATE_OPEN, STATE_HALF_OPEN):
            self._states[backend] = STATE_HEALTHY
            self._last_state_change[backend] = now
            self._half_open_probes[backend] = 0
            self._log_event(backend, prev_state, STATE_HEALTHY, "Successful probe execution")
        elif prev_state == STATE_DEGRADED:
            self._states[backend] = STATE_HEALTHY
            self._last_state_change[backend] = now
            self._log_event(backend, STATE_DEGRADED, STATE_HEALTHY, "Latency returned to normal")

    def record_failure(self, backend: str, reason: str):
        now = time.monotonic()
        prev_state = self.get_state(backend)
        self._consecutive_failures[backend] += 1
        fails = self._consecutive_failures[backend]

        if prev_state == STATE_HALF_OPEN:
            # Probe failed, trip back to open immediately
            self._states[backend] = STATE_OPEN
            self._last_state_change[backend] = now
            self._log_event(backend, STATE_HALF_OPEN, STATE_OPEN, f"Probe failed: {reason}")
        elif fails >= CB_FAILURE_THRESHOLD and prev_state != STATE_OPEN:
            self._states[backend] = STATE_OPEN
            self._last_state_change[backend] = now
            self._log_event(backend, prev_state, STATE_OPEN, f"Exceeded {CB_FAILURE_THRESHOLD} consecutive failures: {reason}")
        elif fails >= 1 and prev_state == STATE_HEALTHY:
            self._states[backend] = STATE_DEGRADED
            self._last_state_change[backend] = now
            self._log_event(backend, STATE_HEALTHY, STATE_DEGRADED, f"Transient failure observed: {reason}")

    def get_observed_ms_per_token(self, backend: str) -> float:
        return self._observed_ms_per_token.get(backend, float(TOKEN_DELAY_MS.get(backend, 100)) * TIME_SCALE)

    def get_status_summary(self) -> Dict[str, Any]:
        return {
            name: {
                "state": self.get_state(name),
                "consecutive_failures": self._consecutive_failures[name],
                "observed_ms_per_token": round(self.get_observed_ms_per_token(name), 2),
                "is_available": self.is_available(name),
            }
            for name in BACKEND_URLS
        }

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._events)

# Global singleton
health_manager = BackendHealthManager()

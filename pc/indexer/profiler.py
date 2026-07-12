"""Phase 1 — profiles which ONNX Runtime execution provider actually serves
each embedding inference (NPU via QNN, GPU via DirectML, or CPU), flags
silent fallback away from the preferred provider, and summarizes backend
usage for debugging and the judging demo.

Input: per-inference (provider, latency_ms) pairs, reported by embedder.py.
Output: `summary()` returns a session-level backend usage report.
Side effects: logs a WARNING on silent fallback, DEBUG otherwise.
"""

import logging
import time
from collections import Counter
from contextlib import contextmanager

logger = logging.getLogger(f"lore.{__name__}")

PROVIDER_TO_BACKEND = {
    "QNNExecutionProvider": "NPU",
    "DmlExecutionProvider": "GPU",
    "CUDAExecutionProvider": "GPU",
    "CPUExecutionProvider": "CPU",
}

# NPU (QNN) -> GPU (DirectML) -> CPU, per CLAUDE.md's explicit fallback order.
DEFAULT_PROVIDER_PREFERENCE = [
    "QNNExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


class InferenceProfiler:
    """Tracks which execution provider actually served each inference call
    against a configured preference order, and reports usage over a session."""

    def __init__(self, preferred_providers=None):
        self.preferred_providers = list(preferred_providers or DEFAULT_PROVIDER_PREFERENCE)
        self._records = []

    def backend_for(self, provider):
        """Map an ONNX Runtime provider name to a human backend label (NPU/GPU/CPU)."""
        return PROVIDER_TO_BACKEND.get(provider, "CPU")

    def record(self, provider, latency_ms):
        """Record one inference call's actual provider and latency.

        Args:
            provider: the ONNX Runtime provider name that actually served
                the call (e.g. session.get_providers()[0]).
            latency_ms: wall-clock inference time in milliseconds.

        Returns:
            The recorded entry: {"provider", "backend", "latency_ms", "fallback"}.
        Side effects: appends to the in-memory session log; logs WARNING if
            the top-preference provider was available in the preference list
            but a lower-priority one actually served this call (silent
            fallback), else logs DEBUG.
        """
        backend = self.backend_for(provider)
        preferred = self.preferred_providers[0] if self.preferred_providers else None
        fell_back = preferred is not None and provider != preferred

        entry = {"provider": provider, "backend": backend, "latency_ms": latency_ms, "fallback": fell_back}
        self._records.append(entry)

        if fell_back:
            logger.warning(
                "ONNX Runtime fell back from %s to %s (backend=%s); latency=%.2fms",
                preferred, provider, backend, latency_ms,
            )
        else:
            logger.debug("Inference served by %s (backend=%s); latency=%.2fms", provider, backend, latency_ms)

        return entry

    @contextmanager
    def track(self, provider):
        """Time the wrapped block and record() it against `provider` automatically."""
        start = time.perf_counter()
        yield
        latency_ms = (time.perf_counter() - start) * 1000
        self.record(provider, latency_ms)

    def summary(self):
        """Return a session summary: per-backend count/percentage, average
        latency, and whether any fallback occurred. Side effects: none."""
        total = len(self._records)
        if total == 0:
            return {"total_inferences": 0, "backends": {}, "any_fallback": False, "avg_latency_ms": 0.0}

        backend_counts = Counter(r["backend"] for r in self._records)
        backends = {
            backend: {"count": count, "percentage": round(count / total * 100, 2)}
            for backend, count in backend_counts.items()
        }
        avg_latency = sum(r["latency_ms"] for r in self._records) / total

        return {
            "total_inferences": total,
            "backends": backends,
            "any_fallback": any(r["fallback"] for r in self._records),
            "avg_latency_ms": round(avg_latency, 3),
        }

    def reset(self):
        """Clear the in-memory session log. Side effects: mutates self._records."""
        self._records.clear()

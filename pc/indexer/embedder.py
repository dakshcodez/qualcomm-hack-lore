"""Phase 1 — embeds text via ONNX Runtime, preferring the QNN (NPU)
execution provider, falling back to DirectML (GPU) then CPU, and reporting
which provider actually served each call through profiler.py.

Real deployment note: on the Snapdragon PC, `model_path` must point at the
QNN-quantized EmbeddingGemma 300M ONNX artifact produced by
scripts/quantize_embedding_model.py, with onnxruntime-qnn installed so
"QNNExecutionProvider" appears in onnxruntime.get_available_providers().
This sandbox has neither the QNN toolchain nor the gated EmbeddingGemma
weights, so the provider-selection/fallback logic below is real and will
run unmodified on the real hardware, but tests exercise it against a tiny
locally-generated synthetic ONNX model instead.

Input: model_path (ONNX file) + a list of raw text strings.
Output: list of embedding vectors (list[float]), one per input text.
Side effects: creates an onnxruntime.InferenceSession (loads the model file
into memory); logs provider selection/fallback via profiler.py.
"""

import logging

import numpy as np
import onnxruntime as ort

from pc.indexer.profiler import DEFAULT_PROVIDER_PREFERENCE, InferenceProfiler

logger = logging.getLogger(__name__)


class ExecutionProviderUnavailableError(RuntimeError):
    """Raised when none of the preferred providers are available in this ONNX Runtime build."""


def _default_preprocess(texts, input_dim):
    """Deterministic bag-of-hashed-words placeholder text encoder.

    NOT a semantic embedding. Phase 1 has no real EmbeddingGemma
    tokenizer/ONNX export available in this sandbox, so this exists purely
    to give embed() a well-defined numeric input to feed the ONNX session.
    Replace with the real tokenizer + model input pipeline once the actual
    EmbeddingGemma ONNX model is deployed on the Snapdragon PC.
    """
    vectors = np.zeros((len(texts), input_dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.split():
            bucket = hash(word) % input_dim
            vectors[i, bucket] += 1.0
    return vectors


class Embedder:
    """Wraps an onnxruntime.InferenceSession with NPU->GPU->CPU provider
    fallback and per-call profiling via InferenceProfiler."""

    def __init__(self, model_path, preferred_providers=None, profiler=None, preprocess_fn=None, session=None):
        self.model_path = model_path
        self.preferred_providers = list(preferred_providers or DEFAULT_PROVIDER_PREFERENCE)
        self.profiler = profiler or InferenceProfiler(preferred_providers=self.preferred_providers)
        self.preprocess_fn = preprocess_fn or _default_preprocess

        self.session = session or self._create_session()
        self.active_provider = self.session.get_providers()[0]
        logger.info("Embedder initialized with active provider=%s", self.active_provider)

    def _create_session(self):
        """Create the ONNX Runtime session using the highest-priority
        available provider(s). Side effects: loads model_path from disk."""
        available = set(ort.get_available_providers())
        requested = [p for p in self.preferred_providers if p in available]
        if not requested:
            raise ExecutionProviderUnavailableError(
                f"None of the preferred providers {self.preferred_providers} are available; "
                f"onnxruntime reports {sorted(available)}"
            )
        return ort.InferenceSession(str(self.model_path), providers=requested)

    def embed(self, texts):
        """Embed a list of raw text strings.

        Args:
            texts: list[str].
        Returns:
            list[list[float]], one embedding vector per input text, in order.
            [] if texts is empty.
        Side effects: runs ONNX Runtime inference; records latency + active
            provider via self.profiler (which logs WARNING on silent
            fallback away from the top-preference provider).
        """
        if not texts:
            return []

        input_meta = self.session.get_inputs()[0]
        input_dim = input_meta.shape[-1]
        if not isinstance(input_dim, int):
            raise ValueError(f"Model input's last dimension must be static, got shape {input_meta.shape}")

        feed = {input_meta.name: self.preprocess_fn(texts, input_dim)}
        output_name = self.session.get_outputs()[0].name

        with self.profiler.track(self.active_provider):
            outputs = self.session.run([output_name], feed)

        return outputs[0].tolist()

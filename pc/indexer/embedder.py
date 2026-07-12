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

import os
import glob
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pc.indexer.profiler import DEFAULT_PROVIDER_PREFERENCE, InferenceProfiler

logger = logging.getLogger(f"lore.{__name__}")


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
        # Find FastRPC DLL on Windows ARM64 dynamically
        fastrpc_dir = None
        try:
            pattern = 'C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll'
            matches = glob.glob(pattern, recursive=True)
            if matches:
                fastrpc_dir = os.path.dirname(matches[0])
        except Exception:
            pass

        # Load QNN Execution Provider if possible
        qnn_registered = False
        provider_options = None
        try:
            import onnxruntime_qnn as qnn_ep
            qnn_dir = os.path.dirname(os.path.abspath(qnn_ep.__file__))
            
            # Set search paths
            paths = [qnn_dir]
            if fastrpc_dir:
                paths.append(fastrpc_dir)
            os.environ['PATH'] = os.pathsep.join(paths) + os.pathsep + os.environ['PATH']
            if hasattr(os, "add_dll_directory"):
                for p in paths:
                    try:
                        os.add_dll_directory(p)
                    except Exception:
                        pass
                        
            # Register library
            ort.register_execution_provider_library('QNNExecutionProvider', qnn_ep.get_library_path())
            qnn_registered = True
            
            # Setup provider options
            provider_options = {
                "QNNExecutionProvider": {"backend_path": qnn_ep.get_qnn_htp_path()}
            }
            logger.info("Successfully registered QNNExecutionProvider with backend_path=%s", qnn_ep.get_qnn_htp_path())
        except Exception as e:
            logger.debug("QNNExecutionProvider registration skipped/failed: %s", e)

        available = set(ort.get_available_providers())
        requested = []
        session_provider_options = []
        
        for provider in self.preferred_providers:
            if provider in available:
                requested.append(provider)
                if provider_options and provider in provider_options:
                    session_provider_options.append(provider_options[provider])
                else:
                    session_provider_options.append({})
                    
        if not requested:
            raise ExecutionProviderUnavailableError(
                f"None of the preferred providers {self.preferred_providers} are available; "
                f"onnxruntime reports {sorted(available)}"
            )
            
        return ort.InferenceSession(
            str(self.model_path), 
            providers=requested,
            provider_options=session_provider_options
        )

    def embed(self, texts, prefix="document: "):
        """Embed a list of raw text strings.

        Args:
            texts: list[str].
            prefix: str. prepended to each text if model expects tokenized inputs.
        Returns:
            list[list[float]], one embedding vector per input text, in order.
            [] if texts is empty.
        Side effects: runs ONNX Runtime inference (one call per batch_size()-sized
            slice of texts — see batch_size()); records latency + active
            provider via self.profiler (which logs WARNING on silent
            fallback away from the top-preference provider) for each such call.
        """
        if not texts:
            return []

        batch_size = self.batch_size()
        if len(texts) <= batch_size:
            return self._embed_batch(texts, prefix)

        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self._embed_batch(texts[i:i + batch_size], prefix))
        return results

    def batch_size(self):
        """The largest number of texts embed() may feed the session in one
        ONNX Runtime call. Models quantized for QNN commonly pin every
        input dimension (including batch) to a static size — feeding more
        texts than that at once raises an ONNX Runtime shape-mismatch error
        (seen as "Got invalid dimensions ... index: 0"), so embed() must
        split into single-batch-sized calls instead when this is fixed.

        Returns:
            The model's static batch dimension (an int) if fixed, else a
            value >= any realistic input length (i.e. "unbounded").
        """
        for inp in self.session.get_inputs():
            if inp.name in ("input_ids", "attention_mask") and inp.shape and isinstance(inp.shape[0], int):
                return inp.shape[0]
        return 2 ** 31

    def _embed_batch(self, texts, prefix):
        """Run one ONNX Runtime inference call over texts (must fit within
        batch_size()). Side effects: as embed()'s, for this one call."""
        # Check model inputs to see if we're in real Gemma mode (expects input_ids/attention_mask)
        # or toy/smoke mode (expects a single vector).
        inputs = self.session.get_inputs()
        input_names = [i.name for i in inputs]

        if "input_ids" in input_names and "attention_mask" in input_names:
            # Gemma mode: load sentencepiece tokenizer if not already done
            if not hasattr(self, "sp_processor"):
                tokenizer_path = Path(self.model_path).parent.parent / "tokenizer.model"
                if not tokenizer_path.exists():
                    # Fallback to model directory itself
                    tokenizer_path = Path(self.model_path).parent / "tokenizer.model"
                if not tokenizer_path.exists():
                    raise FileNotFoundError(f"tokenizer.model not found at {tokenizer_path}")
                
                import sentencepiece as spm
                self.sp_processor = spm.SentencePieceProcessor()
                self.sp_processor.Load(str(tokenizer_path))

            prefixed_texts = [f"{prefix}{t}" for t in texts]
            # Encode with BOS token (Gemma BOS token ID is 2)
            input_ids = [[2] + self.sp_processor.EncodeAsIds(text) for text in prefixed_texts]

            # Check if model has fixed input shape (QNN requires static dims)
            input_id_shape = None
            for inp in inputs:
                if inp.name == "input_ids":
                    input_id_shape = inp.shape
                    break

            # Determine target sequence length
            if input_id_shape and len(input_id_shape) >= 2 and isinstance(input_id_shape[1], int):
                # Fixed-shape model (e.g., QNN-compatible with SEQ_LEN=128)
                target_len = input_id_shape[1]
            else:
                # Dynamic-shape model — pad to longest in batch
                target_len = max(len(seq) for seq in input_ids)

            padded_input_ids = []
            attention_mask = []
            for seq in input_ids:
                if len(seq) > target_len:
                    seq = seq[:target_len]
                pad_len = target_len - len(seq)
                padded_input_ids.append(seq + [0] * pad_len)
                attention_mask.append([1] * len(seq) + [0] * pad_len)

            feed = {
                "input_ids": np.array(padded_input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64)
            }
            # Store attention mask for potential mean-pooling
            attn_mask_np = feed["attention_mask"]
        else:
            # Toy/smoke model mode
            input_meta = inputs[0]
            input_dim = input_meta.shape[-1]
            if not isinstance(input_dim, int):
                raise ValueError(f"Model input's last dimension must be static, got shape {input_meta.shape}")
            feed = {input_meta.name: self.preprocess_fn(texts, input_dim)}
            attn_mask_np = None

        outputs = self.session.get_outputs()
        output_names = [o.name for o in outputs]
        if "sentence_embedding" in output_names:
            output_name = "sentence_embedding"
        else:
            output_name = outputs[0].name

        with self.profiler.track(self.active_provider):
            raw_outputs = self.session.run([output_name], feed)

        result = raw_outputs[0]

        # If output is 3D [batch, seq_len, hidden_dim], apply mean pooling
        # (vanilla ONNX export outputs last_hidden_state, not sentence_embedding)
        if result.ndim == 3 and attn_mask_np is not None:
            # Attention-mask-aware mean pooling: average only non-padding tokens
            mask_expanded = attn_mask_np[:, :, np.newaxis].astype(np.float32)  # [B, S, 1]
            masked = result * mask_expanded  # zero out padding positions
            summed = masked.sum(axis=1)  # [B, hidden]
            counts = mask_expanded.sum(axis=1).clip(min=1e-9)  # [B, 1]
            result = summed / counts  # [B, hidden]

        return result.tolist()

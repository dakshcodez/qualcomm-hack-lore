"""End-to-end pipeline: quantize the vanilla ONNX model to INT8, then test on QNN (NPU).

Steps:
  1. Quantize models/embedding_gemma_300m_vanilla.onnx -> models/embedding_gemma_vanilla_int8.onnx
  2. Load with QNNExecutionProvider
  3. Verify QNN is actually the active provider (not silent CPU fallback)
  4. Benchmark inference latency

Usage:
    python -m pc.scripts.quantize_and_test_npu
"""

import os
import sys
import time
import glob
import json
import numpy as np
from pathlib import Path

VANILLA_FP32 = Path("models/optimum_gemma/model.onnx")
VANILLA_INT8 = Path("models/embedding_gemma_vanilla_int8.onnx")
TOKENIZER = Path("models/tokenizer.model")
CALIBRATION_DIR = Path("models/calibration_data")
SEQ_LEN = 128


def step1_quantize():
    """Quantize the vanilla FP32 model to INT8 using existing infrastructure."""
    if VANILLA_INT8.exists():
        print(f"[Step 1] INT8 model already exists: {VANILLA_INT8}")
        return

    if not VANILLA_FP32.exists():
        print(f"[Step 1] ERROR: FP32 vanilla model not found at {VANILLA_FP32}")
        print("         Run `python -m pc.scripts.export_vanilla_onnx` first.")
        sys.exit(1)

    print(f"[Step 1] Quantizing {VANILLA_FP32} -> {VANILLA_INT8} ...")
    from pc.scripts.quantize_embedding_model import quantize

    quantize(
        input_path=str(VANILLA_FP32),
        calibration_data_dir=str(CALIBRATION_DIR),
        output_path=str(VANILLA_INT8),
        tokenizer_path=str(TOKENIZER),
    )
    print(f"[Step 1] Done. INT8 model: {VANILLA_INT8} ({VANILLA_INT8.stat().st_size / 1e6:.1f} MB)")


def step2_verify_ops():
    """Verify the INT8 model has no com.microsoft ops."""
    import onnx

    print(f"\n[Step 2] Verifying {VANILLA_INT8} has no com.microsoft ops...")
    m = onnx.load(str(VANILLA_INT8))
    ms_ops = [n for n in m.graph.node if n.domain == "com.microsoft"]
    if ms_ops:
        print(f"[Step 2] ❌ FAIL: Found {len(ms_ops)} com.microsoft ops!")
        for n in ms_ops[:5]:
            print(f"         - {n.op_type}")
        sys.exit(1)
    print(f"[Step 2] ✅ PASS: {len(m.graph.node)} nodes, all standard ONNX ops")


def step3_test_npu():
    """Load with QNN EP and verify it's actually the active provider."""
    # -- Locate FastRPC driver --
    fastrpc_dir = None
    try:
        matches = glob.glob('C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll', recursive=True)
        if matches:
            fastrpc_dir = os.path.dirname(matches[0])
            print(f"\n[Step 3] FastRPC driver: {fastrpc_dir}")
    except Exception:
        pass

    # -- Set up QNN paths --
    try:
        import onnxruntime_qnn as qnn_ep
        qnn_dir = os.path.dirname(os.path.abspath(qnn_ep.__file__))
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
    except ImportError:
        print("[Step 3] ❌ onnxruntime_qnn not installed. Cannot test NPU.")
        sys.exit(1)

    import onnxruntime as ort

    ort.register_execution_provider_library('QNNExecutionProvider', qnn_ep.get_library_path())
    print(f"[Step 3] Available providers: {ort.get_available_providers()}")

    ort.set_default_logger_severity(0)
    # -- Create session with profiling --
    opts = ort.SessionOptions()
    opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    opts.enable_profiling = True
    opts.profile_file_prefix = "npu_vanilla_test"
    opts.log_severity_level = 0
    opts.log_verbosity_level = 0

    providers = ['QNNExecutionProvider']
    provider_options = [{'backend_path': qnn_ep.get_qnn_htp_path()}]

    print(f"[Step 3] Loading {VANILLA_INT8}...")
    sess = ort.InferenceSession(
        str(VANILLA_INT8),
        sess_options=opts,
        providers=providers,
        provider_options=provider_options,
    )

    active = sess.get_providers()
    print(f"[Step 3] Active providers: {active}")

    if active[0] == "QNNExecutionProvider":
        print("[Step 3] ✅ QNNExecutionProvider is PRIMARY — running on NPU!")
    else:
        print("[Step 3] ⚠️  QNN is not primary. Checking if it's present at all...")
        if "QNNExecutionProvider" in active:
            print("[Step 3] QNN present but not primary")
        else:
            print("[Step 3] ❌ QNN NOT in active providers — still falling back to CPU")

    # -- Tokenise --
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOKENIZER))
    raw_ids = [2] + sp.EncodeAsIds("document: Lore is a private semantic search engine.")
    if len(raw_ids) > SEQ_LEN:
        raw_ids = raw_ids[:SEQ_LEN]
    pad_len = SEQ_LEN - len(raw_ids)
    input_ids = np.array([raw_ids + [0] * pad_len], dtype=np.int64)
    attn_mask = np.array([[1] * len(raw_ids) + [0] * pad_len], dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attn_mask}

    outputs = sess.get_outputs()
    out_name = outputs[0].name

    # -- Warmup --
    print(f"[Step 3] Running warmup inference...")
    result = sess.run([out_name], feed)
    embedding = result[0]
    print(f"[Step 3] Output shape: {embedding.shape}")
    print(f"[Step 3] First 5 values: {embedding[0][:5]}")

    # -- Benchmark --
    N = 50
    print(f"\n[Step 3] Benchmarking {N} inferences...")
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run([out_name], feed)
    elapsed = time.perf_counter() - t0
    print(f"[Step 3] {N} inferences in {elapsed:.2f}s = {elapsed * 1000 / N:.1f} ms/inference")

    # -- Profile analysis --
    prof = sess.end_profiling()
    try:
        with open(prof, "r") as f:
            events = json.load(f)
        counts = {}
        for ev in events:
            prov = ev.get("args", {}).get("provider")
            if prov:
                counts[prov] = counts.get(prov, 0) + 1
        print(f"\n[Step 3] Nodes per provider:")
        for prov, cnt in sorted(counts.items()):
            print(f"  {prov}: {cnt}")
    finally:
        try:
            os.remove(prof)
        except Exception:
            pass


def main():
    step1_quantize()
    step2_verify_ops()
    step3_test_npu()
    return 0


if __name__ == "__main__":
    sys.exit(main())

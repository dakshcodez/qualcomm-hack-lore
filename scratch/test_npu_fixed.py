"""Test whether the fixed-shape model runs on QNN (NPU) instead of CPU."""
import os
import sys
import time
import glob
import json
import numpy as np

# -- Locate FastRPC driver --
fastrpc_dir = None
try:
    matches = glob.glob('C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll', recursive=True)
    if matches:
        fastrpc_dir = os.path.dirname(matches[0])
        print(f"FastRPC driver: {fastrpc_dir}")
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
            try: os.add_dll_directory(p)
            except: pass
    qnn_available = True
except ImportError:
    qnn_available = False

import onnxruntime as ort

print(f"onnxruntime loaded from: {ort.__file__}")
print(f"onnxruntime version: {ort.__version__}")
print(f"Available providers (before register): {ort.get_available_providers()}")

MODEL = 'models/embedding_gemma_300m_qnn_int8_fixed.onnx'
TOKENIZER = 'models/tokenizer.model'

def main():
    if qnn_available:
        ort.register_execution_provider_library('QNNExecutionProvider', qnn_ep.get_library_path())
        print(f"Available providers (after register): {ort.get_available_providers()}")

    # -- Session with profiling --
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = "npu_test"
    opts.log_severity_level = 0  # Verbose

    providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
    provider_options = [{'backend_path': qnn_ep.get_qnn_htp_path()}, {}]

    print(f"Loading {MODEL}...")
    sess = ort.InferenceSession(MODEL, sess_options=opts,
                                providers=providers,
                                provider_options=provider_options)
    active = sess.get_providers()
    print(f"Active providers: {active}")

    # -- Tokenise --
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(); sp.Load(TOKENIZER)
    raw_ids = [2] + sp.EncodeAsIds("document: Lore is a private semantic search engine.")
    SEQ_LEN = 128
    if len(raw_ids) > SEQ_LEN:
        raw_ids = raw_ids[:SEQ_LEN]
    pad_len = SEQ_LEN - len(raw_ids)
    input_ids = np.array([raw_ids + [0]*pad_len], dtype=np.int64)
    attn_mask = np.array([[1]*len(raw_ids) + [0]*pad_len], dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attn_mask}
    out_name = sess.get_outputs()[0].name

    # -- Warmup + benchmark --
    sess.run([out_name], feed)
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run([out_name], feed)
    elapsed = time.perf_counter() - t0
    print(f"{N} inferences in {elapsed:.2f}s = {elapsed*1000/N:.1f} ms/inference")

    # -- Profile analysis --
    prof = sess.end_profiling()
    with open(prof, "r") as f:
        events = json.load(f)
    counts = {}
    for ev in events:
        prov = ev.get("args", {}).get("provider")
        if prov:
            counts[prov] = counts.get(prov, 0) + 1
    print("Nodes per provider:")
    for prov, cnt in counts.items():
        print(f"  {prov}: {cnt}")
    os.remove(prof)

if __name__ == "__main__":
    main()

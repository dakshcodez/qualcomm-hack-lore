import os
import sys
import time
import glob
import numpy as np

# Find FastRPC DLL on Windows ARM64 dynamically
fastrpc_dir = None
try:
    pattern = 'C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll'
    matches = glob.glob(pattern, recursive=True)
    if matches:
        fastrpc_dir = os.path.dirname(matches[0])
        print(f"Located FastRPC driver directory: {fastrpc_dir}")
except Exception as e:
    print(f"Warning searching for FastRPC: {e}")

# Resolve QNN binaries and set PATH/DLL Directories
try:
    import onnxruntime_qnn as qnn_ep
    qnn_dir = os.path.dirname(os.path.abspath(qnn_ep.__file__))
    paths_to_add = [qnn_dir]
    if fastrpc_dir:
        paths_to_add.append(fastrpc_dir)
        
    os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + os.environ['PATH']
    
    if hasattr(os, "add_dll_directory"):
        for p in paths_to_add:
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
    qnn_available = True
except ImportError:
    qnn_available = False

import onnxruntime as ort

def run_test():
    model_path = 'models/embedding_gemma_300m_qnn_int8.onnx'
    tokenizer_path = 'models/tokenizer.model'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("Error: Required model or tokenizer files are missing.")
        sys.exit(1)
        
    print("Initializing ONNX Runtime session...")
    
    if qnn_available:
        print("Registering QNN Execution Provider plugin...")
        ort.register_execution_provider_library('QNNExecutionProvider', qnn_ep.get_library_path())
        
        # Load session with QNN and CPU backup
        providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {'backend_path': qnn_ep.get_qnn_htp_path()},
            {}
        ]
    else:
        print("Warning: onnxruntime_qnn helper not available, using standard providers.")
        providers = ['CPUExecutionProvider']
        provider_options = [{}]
        
    sess = ort.InferenceSession(
        model_path,
        providers=providers,
        provider_options=provider_options
    )
    
    print("Session active providers:", sess.get_providers())
    
    # Tokenize input
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    ids = [[2] + sp.EncodeAsIds("document: Lore is a private semantic search engine.")]
    max_len = len(ids[0])
    input_ids = np.array(ids, dtype=np.int64)
    attention_mask = np.array([[1] * max_len], dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    output_name = sess.get_outputs()[0].name
    
    print("Running 500 inferences to check NPU utilization...")
    # Warm up
    sess.run([output_name], feed)
    
    start = time.perf_counter()
    for _ in range(500):
        sess.run([output_name], feed)
    elapsed = time.perf_counter() - start
    
    print(f"Test finished in {elapsed:.2f} seconds.")
    print(f"Average latency per inference: {elapsed * 1000 / 500:.2f} ms")

if __name__ == "__main__":
    run_test()

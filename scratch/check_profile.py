import os
import sys
import glob
import json
import numpy as np

# Find FastRPC DLL on Windows ARM64 dynamically
fastrpc_dir = None
try:
    pattern = 'C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll'
    matches = glob.glob(pattern, recursive=True)
    if matches:
        fastrpc_dir = os.path.dirname(matches[0])
except Exception:
    pass

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

def check_profile():
    model_path = 'models/embedding_gemma_300m_qnn_int8.onnx'
    tokenizer_path = 'models/tokenizer.model'
    
    if qnn_available:
        ort.register_execution_provider_library('QNNExecutionProvider', qnn_ep.get_library_path())
        providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{'backend_path': qnn_ep.get_qnn_htp_path()}, {}]
    else:
        providers = ['CPUExecutionProvider']
        provider_options = [{}]
        
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = "ort_profile"
    
    sess = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers,
        provider_options=provider_options
    )
    
    # Run 1 inference
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    ids = [[2] + sp.EncodeAsIds("document: Lore is a private semantic search engine.")]
    max_len = len(ids[0])
    input_ids = np.array(ids, dtype=np.int64)
    attention_mask = np.array([[1] * max_len], dtype=np.int64)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    output_name = sess.get_outputs()[0].name
    
    sess.run([output_name], feed)
    
    # End profiling and get file path
    profile_path = sess.end_profiling()
    print(f"Profile saved to: {profile_path}")
    
    # Parse profile
    with open(profile_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    # Count occurrences of providers in trace nodes
    provider_counts = {}
    for event in trace_data:
        if "args" in event and "provider" in event["args"]:
            prov = event["args"]["provider"]
            provider_counts[prov] = provider_counts.get(prov, 0) + 1
            
    print("Node Execution Counts per Provider:")
    for prov, count in provider_counts.items():
        print(f"- {prov}: {count} node(s)")
        
    # Clean up profile file
    try:
        os.remove(profile_path)
    except Exception:
        pass

if __name__ == "__main__":
    check_profile()

"""Decompose com.microsoft fused ops into standard ONNX ops for QNN compatibility.

The community ONNX export of EmbeddingGemma uses two Microsoft-specific fused
operators that QNN cannot run on the NPU:
  - com.microsoft::MultiHeadAttention
  - com.microsoft::RotaryEmbedding

This script replaces them with standard ONNX ops (MatMul, Softmax, Add, etc.)
so QNN can partition the entire graph onto the Hexagon NPU.

We use onnxruntime's built-in graph optimization to do this:
ORT can inline contrib ops when we set optimization level to DISABLE_ALL and
use the onnx_shape_inference pass. But the real trick is to use Olive or
export from PyTorch directly without fused ops.

Alternative approach: Use onnxruntime's graph optimization with the QNN EP
context binary generation, which handles decomposition internally.
"""
import os
import sys

def try_olive_approach():
    """Use Microsoft Olive to optimize the model for QNN."""
    print("Attempting Olive-based optimization...")
    try:
        from olive.workflows import run as olive_run
        print("Olive is available - this would be the best approach")
        return True
    except ImportError:
        print("Olive not installed. Trying alternative approach.")
        return False


def try_ort_optimization():
    """Use ORT's built-in optimization to generate a QNN context binary.
    
    When creating a session with QNN EP, ORT can generate a 'context binary'
    (.onnx with embedded QNN graph) that pre-compiles the model for the NPU.
    This handles op decomposition internally.
    """
    import glob
    import onnxruntime as ort

    # Setup QNN paths
    fastrpc_dir = None
    try:
        matches = glob.glob('C:/Windows/System32/DriverStore/FileRepository/**/libcdsprpc.dll', recursive=True)
        if matches:
            fastrpc_dir = os.path.dirname(matches[0])
    except Exception:
        pass

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

    ort.register_execution_provider_library('QNNExecutionProvider', qnn_ep.get_library_path())

    input_model = 'models/embedding_gemma_300m_qnn_int8_fixed.onnx'
    output_model = 'models/embedding_gemma_qnn_context.onnx'

    print(f"Generating QNN context binary from {input_model}...")
    print("This pre-compiles the model for the Hexagon NPU (may take several minutes)...")
    
    opts = ort.SessionOptions()
    # Enable context binary generation
    opts.add_session_config_entry("ep.context_enable", "1")
    opts.add_session_config_entry("ep.context_file_path", output_model)
    # Set graph optimization level
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        sess = ort.InferenceSession(
            input_model,
            sess_options=opts,
            providers=['QNNExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[
                {
                    'backend_path': qnn_ep.get_qnn_htp_path(),
                    'htp_performance_mode': 'burst',
                    'htp_graph_finalization_optimization_mode': '3',
                },
                {}
            ]
        )
        print(f"Active providers: {sess.get_providers()}")
        
        if os.path.exists(output_model):
            size_mb = os.path.getsize(output_model) / (1024*1024)
            print(f"Context binary generated: {output_model} ({size_mb:.1f} MB)")
            return True
        else:
            print("Context binary was not generated.")
            return False
    except Exception as e:
        print(f"Context binary generation failed: {e}")
        return False


def try_direct_pytorch_export():
    """Re-export from PyTorch without fused Microsoft ops.
    
    This requires torch + transformers, which may not be installed.
    """
    print("\nAttempting direct PyTorch re-export without fused ops...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        print("PyTorch and transformers available.")
        
        model_name = "google/embeddinggemma-300m"
        print(f"Loading {model_name}...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Create dummy inputs with fixed shapes
        dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)
        
        output_path = "models/embedding_gemma_300m_vanilla.onnx"
        print(f"Exporting to {output_path} (no fused ops)...")
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            opset_version=17,  # Standard ONNX opset, no contrib ops
            do_constant_folding=True,
        )
        print(f"Export complete: {output_path}")
        return True
    except ImportError as e:
        print(f"PyTorch/transformers not available: {e}")
        return False
    except Exception as e:
        print(f"Export failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Decomposing Microsoft fused ops for QNN NPU compatibility")
    print("=" * 60)
    
    # Try context binary generation first (uses existing model)
    success = try_ort_optimization()
    
    if not success:
        print("\nContext binary generation failed.")
        print("Trying PyTorch re-export approach...")
        success = try_direct_pytorch_export()
    
    if not success:
        print("\n" + "=" * 60)
        print("SUMMARY: Neither approach succeeded automatically.")
        print("Manual steps needed:")
        print("1. Install olive-ai: pip install olive-ai")
        print("2. Or install torch + transformers for re-export")
        print("3. Or use Qualcomm AI Hub (https://aihub.qualcomm.com/)")
        print("=" * 60)

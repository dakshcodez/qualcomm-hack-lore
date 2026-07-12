"""Re-export EmbeddingGemma 300M from HuggingFace PyTorch weights to ONNX
using ONLY standard ONNX operators (opset 17).

The pre-exported ONNX from onnx-community used com.microsoft fused ops
(MultiHeadAttention, RotaryEmbedding) which the QNN Execution Provider
does not support, causing 100% silent fallback to CPU.

This script uses torch.onnx.export with the vanilla PyTorch forward pass,
which decomposes attention into standard MatMul/Add/Softmax/etc. that QNN
*does* support.

Usage:
    python -m pc.scripts.export_vanilla_onnx

Output:
    models/embedding_gemma_300m_vanilla.onnx
"""

import sys
from pathlib import Path

MODEL_ID = "google/embeddinggemma-300m"
OUTPUT_PATH = Path("models/embedding_gemma_300m_vanilla.onnx")
SEQ_LEN = 128  # Fixed sequence length for QNN (no dynamic shapes)


def export():
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model: {MODEL_ID}")
    # Use attn_implementation="eager" to avoid SDPA (Scaled Dot Product Attention)
    # custom autograd functions that cause RuntimeError during TorchScript tracing.
    base_model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model.eval()

    # Wrapper to give torch.onnx.export a clean forward(input_ids, attention_mask)
    # signature. The raw Gemma3TextModel.forward() has complex kwargs (use_cache,
    # past_key_values, etc.) that conflict with TorchScript tracing.
    class EmbeddingWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )
            # out is a tuple; first element is last_hidden_state [B, S, D]
            return out[0]

    wrapper = EmbeddingWrapper(base_model)
    wrapper.eval()

    # Fixed shapes for QNN — all input dims must be static integers
    dummy_input_ids = torch.ones(1, SEQ_LEN, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {OUTPUT_PATH} with opset 17 (standard ONNX only)...")

    # Use dynamo=False to force legacy TorchScript-based export.
    # PyTorch 2.13+ defaults to dynamo/torch.export which can't handle
    # DynamicCache in the model output.
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(OUTPUT_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            opset_version=17,  # Standard ONNX only, no com.microsoft ops
            do_constant_folding=True,
            dynamo=False,  # Force legacy TorchScript-based export
        )

    print(f"Export complete: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")

    # Verify no com.microsoft ops remain
    verify()


def verify():
    import onnx

    print(f"\nVerifying {OUTPUT_PATH}...")
    m = onnx.load(str(OUTPUT_PATH))

    all_domains = set()
    ms_ops = []
    op_counts = {}

    for n in m.graph.node:
        domain = n.domain or "onnx"
        all_domains.add(domain)
        key = f"{domain}::{n.op_type}"
        op_counts[key] = op_counts.get(key, 0) + 1
        if n.domain == "com.microsoft":
            ms_ops.append(n.op_type)

    print(f"Total nodes: {len(m.graph.node)}")
    print(f"Domains used: {sorted(all_domains)}")
    print(f"\nOp breakdown:")
    for key in sorted(op_counts.keys()):
        print(f"  {key}: {op_counts[key]}")

    if ms_ops:
        print(f"\n❌ FAIL: Found {len(ms_ops)} com.microsoft ops: {set(ms_ops)}")
        return False
    else:
        print(f"\n✅ PASS: No com.microsoft ops — QNN-compatible!")
        return True


def main():
    export()
    return 0


if __name__ == "__main__":
    sys.exit(main())

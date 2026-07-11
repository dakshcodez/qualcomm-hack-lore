"""Utility — builds a tiny synthetic ONNX embedding model for smoke-testing
the indexing/API pipeline (session creation, execution-provider selection,
LanceDB wiring) before a real EmbeddingGemma 300M ONNX export is available.

NOT a real embedding model: it's a single MatMul against random weights, so
its output vectors are semantically meaningless (see embedder.py's
placeholder bag-of-hashed-words preprocessing). Useful only for verifying
that onnxruntime, the execution providers (NPU/GPU/CPU), and the rest of
the pipeline actually work end-to-end on a given machine.

Usage:
    python pc/scripts/build_smoke_model.py --output smoke_model.onnx
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

DEFAULT_INPUT_DIM = 768
DEFAULT_EMBEDDING_DIM = 768


def build_smoke_model(output_path, input_dim=DEFAULT_INPUT_DIM, embedding_dim=DEFAULT_EMBEDDING_DIM, seed=0):
    """Write a single-MatMul ONNX graph (input "input" [None, input_dim] ->
    output "embedding" [None, embedding_dim]) to output_path.

    Returns:
        output_path, unchanged.
    Side effects: writes output_path.
    """
    weight = np.random.RandomState(seed).randn(input_dim, embedding_dim).astype(np.float32)
    weight_initializer = helper.make_tensor(
        "weight", TensorProto.FLOAT, weight.shape, weight.flatten().tolist()
    )
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_dim])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, embedding_dim])
    node = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["embedding"])
    graph = helper.make_graph(
        [node], "smoke_embedder", [input_tensor], [output_tensor], initializer=[weight_initializer]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build a tiny synthetic ONNX model for pipeline smoke-testing."
    )
    parser.add_argument("--output", required=True, help="Path to write the .onnx model.")
    parser.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    output_path = build_smoke_model(args.output, args.input_dim, args.embedding_dim)
    print(f"Smoke-test model written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

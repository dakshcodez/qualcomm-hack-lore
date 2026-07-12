"""Fix dynamic shapes in the quantized ONNX model so QNN EP can accept it.

QNN requires ALL dimensions to be static integers. Our model has:
  input_ids:      [batch_size, total_sequence_length]
  attention_mask: [batch_size, total_sequence_length]

We fix them to [1, 128] — batch=1 (we embed one doc at a time),
seq_len=128 (padded/truncated to this fixed length at inference time).
"""
import onnx
from onnx import TensorShapeProto

INPUT_MODEL  = "models/embedding_gemma_300m_qnn_int8.onnx"
OUTPUT_MODEL = "models/embedding_gemma_300m_qnn_int8_fixed.onnx"

FIXED_DIMS = {
    "batch_size": 1,
    "total_sequence_length": 128,
}

def fix_dim(dim, dim_map):
    """Fix a single TensorShapeProto.Dimension in-place."""
    if dim.HasField("dim_param") and dim.dim_param in dim_map:
        old_name = dim.dim_param
        new_val = dim_map[old_name]
        # Clear the dim_param by using ClearField, then set dim_value
        dim.ClearField("dim_param")
        dim.dim_value = new_val
        return old_name, new_val
    return None, None

def fix_shape(shape, dim_map, label=""):
    if shape is None:
        return
    for dim in shape.dim:
        old, new = fix_dim(dim, dim_map)
        if old is not None:
            print(f"  {label}: '{old}' -> {new}")

def main():
    print(f"Loading {INPUT_MODEL}...")
    model = onnx.load(INPUT_MODEL)

    # Fix graph inputs
    for inp in model.graph.input:
        if inp.type.tensor_type.HasField("shape"):
            fix_shape(inp.type.tensor_type.shape, FIXED_DIMS, inp.name)

    # Fix graph outputs
    for out in model.graph.output:
        if out.type.tensor_type.HasField("shape"):
            fix_shape(out.type.tensor_type.shape, FIXED_DIMS, f"output:{out.name}")

    # Fix value_info (intermediate tensors)
    for vi in model.graph.value_info:
        if vi.type.tensor_type.HasField("shape"):
            fix_shape(vi.type.tensor_type.shape, FIXED_DIMS, f"vi:{vi.name}")

    print(f"\nSaving fixed model to {OUTPUT_MODEL}...")
    onnx.save(model, OUTPUT_MODEL)

    # Verify
    print("\nVerification:")
    m2 = onnx.load(OUTPUT_MODEL)
    for inp in m2.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
        print(f"  {inp.name}: {dims}")
    for out in m2.graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
        print(f"  output:{out.name}: {dims}")
    print("Done.")

if __name__ == "__main__":
    main()

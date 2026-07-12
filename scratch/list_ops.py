import onnx
m = onnx.load('models/embedding_gemma_300m_qnn_int8_fixed.onnx')
ops = set()
for n in m.graph.node:
    ops.add((n.op_type, n.domain or "onnx"))
print(f"Total unique operator types: {len(ops)}")
for op, domain in sorted(ops):
    print(f"  {domain}::{op}")

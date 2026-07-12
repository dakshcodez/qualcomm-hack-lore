import onnx

def make_shapes_static(model_path, output_path, batch_size=1, seq_len=128):
    model = onnx.load(model_path)
    
    # Update input shapes
    for input_node in model.graph.input:
        dim = input_node.type.tensor_type.shape.dim
        if len(dim) >= 2:
            dim[0].dim_value = batch_size
            dim[1].dim_value = seq_len
            
    # Update output shapes (usually [batch_size, seq_len, hidden_size])
    for output_node in model.graph.output:
        dim = output_node.type.tensor_type.shape.dim
        if len(dim) >= 2:
            dim[0].dim_value = batch_size
            dim[1].dim_value = seq_len
            
    # Remove shape inference info inside the graph that might conflict
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, output_path)
    print(f"Fixed shapes to B={batch_size}, S={seq_len} and saved to {output_path}")

if __name__ == "__main__":
    make_shapes_static("models/optimum_gemma/model.onnx", "models/optimum_gemma/model.onnx")

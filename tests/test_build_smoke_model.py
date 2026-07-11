import numpy as np
import onnxruntime as ort
import pytest

from pc.scripts.build_smoke_model import build_arg_parser, build_smoke_model, main


def test_build_smoke_model_produces_a_loadable_model(tmp_path):
    output_path = tmp_path / "smoke.onnx"

    result_path = build_smoke_model(output_path, input_dim=16, embedding_dim=8)

    assert result_path == output_path
    assert output_path.exists()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(3, 16).astype(np.float32)
    outputs = session.run(None, {"input": x})
    assert outputs[0].shape == (3, 8)


def test_build_smoke_model_uses_default_dims(tmp_path):
    output_path = build_smoke_model(tmp_path / "smoke.onnx")
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [None, 768]
    assert session.get_outputs()[0].shape == [None, 768]


def test_build_smoke_model_is_deterministic_for_a_given_seed(tmp_path):
    path1 = build_smoke_model(tmp_path / "a.onnx", input_dim=16, embedding_dim=8, seed=1)
    path2 = build_smoke_model(tmp_path / "b.onnx", input_dim=16, embedding_dim=8, seed=1)

    session1 = ort.InferenceSession(str(path1), providers=["CPUExecutionProvider"])
    session2 = ort.InferenceSession(str(path2), providers=["CPUExecutionProvider"])
    x = np.random.rand(2, 16).astype(np.float32)

    out1 = session1.run(None, {"input": x})[0]
    out2 = session2.run(None, {"input": x})[0]
    assert np.allclose(out1, out2)


def test_arg_parser_requires_output():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_writes_model_and_prints_confirmation(tmp_path, capsys):
    output_path = tmp_path / "smoke.onnx"
    exit_code = main(["--output", str(output_path), "--input-dim", "4", "--embedding-dim", "2"])

    assert exit_code == 0
    assert output_path.exists()
    assert "Smoke-test model written to" in capsys.readouterr().out

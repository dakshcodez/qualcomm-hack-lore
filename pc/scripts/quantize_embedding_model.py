"""Phase 1 — standalone QNN-targeted static quantization script for the
embedding model (EmbeddingGemma 300M ONNX export) used by embedder.py.

Not part of the live indexing pipeline — run manually/offline to produce a
quantized .onnx artifact for on-NPU inference via the QNN execution
provider.

Real deployment note: this sandbox has no QNN toolchain and no real
EmbeddingGemma ONNX export, so quantization here is validated structurally
(CLI parsing, calibration-data loading, static quantization running
end-to-end) against a tiny synthetic ONNX model in tests. Real INT8/QNN-
hardware-compatibility validation must happen on the actual QNN toolchain
on the Snapdragon PC.

Calibration data contract: --calibration-data points at a directory of
.npy files. For Gemma, if the directory is empty, it will automatically
auto-generate calibration data using the Gemma tokenizer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class DirectoryCalibrationDataReader(CalibrationDataReader):
    """Feeds calibration tensors loaded from .npy files in a directory to
    onnxruntime.quantization.quantize_static, one file per calibration step."""

    def __init__(self, calibration_dir, input_infos, tokenizer_path=None, max_samples=None):
        self.input_infos = input_infos
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        input_names = [info[0] for info in input_infos]

        # Check if we need to auto-generate calibration data for Gemma
        if "input_ids" in input_names and "attention_mask" in input_names:
            existing_npy = list(self.calibration_dir.glob("*.npy"))
            if not existing_npy:
                print("No calibration files found. Auto-generating calibration data...")
                if tokenizer_path is None or not Path(tokenizer_path).exists():
                    raise FileNotFoundError(
                        f"tokenizer.model not found at {tokenizer_path}. "
                        "Please pass --tokenizer to specify the path to tokenizer.model."
                    )
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(str(tokenizer_path))

                sentences = [
                    "What is the Snapchat Multiverse Hackathon?",
                    "Lore is a private semantic memory system.",
                    "Qualcomm Cloud AI 100 handles generation.",
                    "EmbeddingGemma 300M is running on the NPU.",
                    "This system watches the Windows user folder.",
                    "A browser extension tracks the dwell time.",
                    "The phone communicates over WiFi.",
                    "All computations happen on owned hardware.",
                    "Semantic embeddings are stored in LanceDB.",
                    "Sarvam AI provides voice translation support.",
                    "The system indexes PDFs, DOCX, PPTX, and XLSX files.",
                    "FastAPI serves queries on the local network.",
                    "ONNX Runtime executes the model on the Snapdragon NPU.",
                    "Euclidean distance search is used for nearest neighbors.",
                    "Llama 3 or Phi-3 generates responses on Cloud AI 100.",
                    "The Android interface displays query answers and sources."
                ]
                prefixed_sentences = [f"document: {s}" for s in sentences]
                for idx, text in enumerate(prefixed_sentences):
                    ids = [2] + sp.EncodeAsIds(text)
                    if len(ids) > 128:
                        ids = ids[:128]
                    pad_len = 128 - len(ids)
                    ids = ids + [0] * pad_len
                    mask = [1] * (128 - pad_len) + [0] * pad_len
                    np.save(self.calibration_dir / f"sample_{idx}_input_ids.npy", np.array([ids], dtype=np.int64))
                    np.save(self.calibration_dir / f"sample_{idx}_attention_mask.npy", np.array([mask], dtype=np.int64))

        # Load samples
        self._samples = []
        if "input_ids" in input_names and "attention_mask" in input_names:
            input_ids_paths = sorted(self.calibration_dir.glob("*_input_ids.npy"))
            if max_samples is not None:
                input_ids_paths = input_ids_paths[:max_samples]
            for path in input_ids_paths:
                mask_path = path.parent / path.name.replace("_input_ids.npy", "_attention_mask.npy")
                if mask_path.exists():
                    self._samples.append({
                        "input_ids": np.load(path).astype(np.int64),
                        "attention_mask": np.load(mask_path).astype(np.int64)
                    })
        else:
            paths = sorted(self.calibration_dir.glob("*.npy"))
            if max_samples is not None:
                paths = paths[:max_samples]
            input_name, input_type = input_infos[0]
            dtype = np.float32 if input_type == 1 else np.int64
            for path in paths:
                arr = np.load(path).astype(dtype)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                self._samples.append({input_name: arr})

        self._iterator = iter(self._samples)

    def get_next(self):
        return next(self._iterator, None)

    def rewind(self):
        self._iterator = iter(self._samples)


def get_model_inputs(model_path):
    """Return model input names and element types from an ONNX model file."""
    model = onnx.load(str(model_path))
    return [(i.name, i.type.tensor_type.elem_type) for i in model.graph.input]


def get_model_input_name(model_path):
    """Return the first input tensor's name from an ONNX model file (maintained for backward compatibility)."""
    return get_model_inputs(model_path)[0][0]


def quantize(input_path, calibration_data_dir, output_path, max_calibration_samples=None, tokenizer_path=None):
    """Run QNN-targeted static INT8 quantization (QDQ format, per-tensor
    QUInt8 activations / QInt8 weights — widely supported by QNN EP kernels).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    calibration_dir = Path(calibration_data_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    input_infos = get_model_inputs(input_path)
    input_names = [info[0] for info in input_infos]

    if "input_ids" in input_names and "attention_mask" in input_names:
        if tokenizer_path is None:
            tokenizer_path = input_path.parent.parent / "tokenizer.model"
            if not tokenizer_path.exists():
                tokenizer_path = input_path.parent / "tokenizer.model"
    else:
        # For non-Gemma models, verify calibration directory exists
        if not calibration_dir.is_dir():
            raise FileNotFoundError(f"Calibration data directory not found: {calibration_dir}")

    calibration_reader = DirectoryCalibrationDataReader(
        calibration_dir, input_infos, tokenizer_path, max_calibration_samples
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="QNN-targeted static INT8 quantization for the embedding ONNX model."
    )
    parser.add_argument("--input", required=True, help="Path to the source .onnx model (fp32).")
    parser.add_argument("--calibration-data", required=True, help="Directory of .npy calibration tensors.")
    parser.add_argument("--output", required=True, help="Path to write the quantized .onnx artifact.")
    parser.add_argument(
        "--max-calibration-samples",
        type=int,
        default=None,
        help="Optional cap on the number of calibration files to use.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.model (required to auto-generate calibration data for Gemma).",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        output_path = quantize(
            args.input, args.calibration_data, args.output, args.max_calibration_samples, args.tokenizer
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Quantized model written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

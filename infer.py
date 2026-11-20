"""Reconstruct a 3-D wind field from a prepared DeepWind-LiDAR input."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "deepwind-lidar.onnx"
INPUT_SHAPE = (19, 300, 300)
OUTPUT_SHAPE = (1, 4, 27, 300, 300)
VARIABLES = ("u", "v", "w", "k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prepared 19-channel .npy file")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the ONNX model")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads (default: 4)")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    if not args.input.is_file() or args.input.suffix.lower() != ".npy":
        parser.error(f"Expected an existing .npy input: {args.input}")

    features = np.load(args.input, allow_pickle=False)
    if features.shape != INPUT_SHAPE:
        parser.error(f"Expected input shape {INPUT_SHAPE}, got {features.shape}")
    if features.dtype.kind not in "fiu":
        parser.error(f"Expected a real numeric input, got {features.dtype}")
    features = np.ascontiguousarray(features[None, ...], dtype=np.float32)
    if not np.isfinite(features).all():
        parser.error("Input contains non-finite values")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    prediction, = session.run(["wind_field"], {"lidar_columns": features})
    if prediction.shape != OUTPUT_SHAPE or not np.isfinite(prediction).all():
        raise RuntimeError("The model returned an invalid wind field.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.savez_compressed(output, **dict(zip(VARIABLES, prediction[0])))
    print(f"Saved u, v, w and k to {args.output}")


if __name__ == "__main__":
    main()

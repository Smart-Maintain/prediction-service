from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prediction_service import model as model_module  # noqa: E402
from prediction_service.paths import MODELS_DIR, ONNX_DIR  # noqa: E402


def main() -> None:
    print("Starting Step 8: Export Model to ONNX...")

    device = torch.device("cpu")
    model = model_module.MultiTaskModel(n_features=14, window_size=30).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_multitask_model.pth", map_location=device))
    model.eval()

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_file_path = ONNX_DIR / "multitask_model.onnx"
    dummy_input = torch.randn(1, 30, 14).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["rul_output", "alert_output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "rul_output": {0: "batch_size"},
            "alert_output": {0: "batch_size"},
        },
    )

    print(f"Successfully exported PyTorch model to ONNX format: {onnx_file_path}")


if __name__ == "__main__":
    main()

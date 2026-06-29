from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = REPO_ROOT / "figures"


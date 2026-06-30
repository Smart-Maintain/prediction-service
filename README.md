# Aerospace Predictive Maintenance AI Module

This repository contains the AI work for an aerospace predictive maintenance system. The project predicts the Remaining Useful Life (RUL) of turbofan engines and returns a failure-alert probability that backend and frontend teams can use inside the main application.

The work is based on the NASA C-MAPSS turbofan degradation dataset. The final deployable artifact is a FastAPI inference service that loads an exported ONNX model and exposes a `/predict` endpoint.

## What Was Built

The project covers the full machine learning workflow:

1. Exploratory data analysis on C-MAPSS FD001, FD002, FD003, and FD004.
2. Sensor cleanup and normalization.
3. RUL regression label generation.
4. Binary alert label generation for engines with `RUL <= 30` cycles.
5. Sliding-window sequence generation using the last 30 engine cycles.
6. A multi-task Bi-LSTM model with two outputs:
   - predicted Remaining Useful Life in cycles
   - failure alert probability
7. Training with validation metrics, early stopping, learning-rate scheduling, and gradient clipping.
8. Evaluation with RMSE, NASA score, F1, ROC AUC, PR AUC, RUL plots, and feature-importance analysis.
9. Export of the trained PyTorch model to ONNX for lightweight backend inference.
10. A FastAPI API wrapper for deployment and integration.

## Repository Layout

```text
back-aerospace-predictive-maintenance/
|-- README.md
|-- issue_report.md
`-- ai study/
    `-- work/
        |-- 01_eda.py
        |-- 02_preprocessing.py
        |-- 03_labels.py
        |-- 04_sequences.py
        |-- 05_model.py
        |-- 06_train.py
        |-- 07_evaluate.py
        |-- 08_export_onnx.py
        |-- best_multitask_model.pth
        |-- *_preprocessed.csv
        |-- *_labeled.csv
        |-- *.png
        `-- api/
            |-- main.py
            |-- requirements.txt
            |-- Dockerfile
            |-- multitask_model.onnx
            `-- multitask_model.onnx.data
```

The `api/` folder is the part backend/devops needs for deployment. The training scripts and generated CSV files are useful for retraining, debugging, and explaining how the model was created.

## Technologies Used

### Machine Learning and Data Processing

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Seaborn
- SHAP / permutation-style feature importance

### Model Deployment

- ONNX
- ONNX Runtime
- FastAPI
- Pydantic
- Uvicorn
- Docker

## Dataset

The model uses NASA C-MAPSS turbofan run-to-failure data:

- FD001: one operating condition, one fault mode
- FD002: six operating conditions, one fault mode
- FD003: one operating condition, two fault modes
- FD004: six operating conditions, two fault modes

Each row represents one engine cycle. The raw files contain:

- engine unit number
- cycle number
- 3 operating settings
- 21 sensor measurements

The project drops weak or redundant sensors and keeps 14 final input features for the model.

## Preprocessing Summary

The preprocessing pipeline in `02_preprocessing.py`:

- loads train and test files for FD001-FD004
- drops sensors `s1`, `s5`, `s6`, `s10`, `s14`, `s16`, `s18`, `s19`, `s20`, and `s21`
- keeps the remaining useful sensors
- scales sensor values with `MinMaxScaler`
- fits scalers only on training data to avoid data leakage
- handles FD002 and FD004 with 6 operating-condition clusters using KMeans
- sorts each engine by `unit` and `cycle`
- writes `*_preprocessed.csv` files

The label pipeline in `03_labels.py`:

- calculates RUL as the number of cycles remaining before failure
- caps RUL at `125`
- creates the binary alert target with this rule:

```text
alert = 1 when RUL <= 30
alert = 0 when RUL > 30
```

The sequence pipeline in `04_sequences.py` builds sliding windows:

```text
input shape = (batch, 30 cycles, 14 features)
```

## Final Model

The model is defined in `05_model.py`.

Architecture:

- 2-layer bidirectional LSTM
- input size: 14 features
- window size: 30 cycles
- hidden size: 64 per direction
- shared representation: 128 dimensions
- dropout: 0.3
- RUL regression head: `Linear(128 -> 64) -> ReLU -> Linear(64 -> 1)`
- alert classification head: `Linear(128 -> 64) -> ReLU -> Linear(64 -> 1) -> Sigmoid`

The model is multi-task because it predicts both:

- `rul_cycles`: remaining useful life
- `alert`: whether the engine is close to failure

Training in `06_train.py` uses:

- Adam optimizer
- weight decay
- `ReduceLROnPlateau`
- early stopping
- combined loss: `MSE(RUL) + 100 * BCE(alert)`
- gradient clipping with `max_norm=1.0`

The trained PyTorch weights are saved as:

```text
ai study/work/best_multitask_model.pth
```

The exported inference model is saved as:

```text
ai study/work/api/multitask_model.onnx
ai study/work/api/multitask_model.onnx.data
```

Keep both ONNX files together. The `.onnx` file references `multitask_model.onnx.data`.

## API

The deployable API is implemented in:

```text
ai study/work/api/main.py
```

It uses FastAPI and ONNX Runtime. PyTorch is not required in the deployed backend container.

### Endpoint

```http
POST /predict
```

Swagger UI is available when the API is running:

```text
http://localhost:8000/docs
```

### Request Body

The API expects one sliding window for one engine:

```json
{
  "window": [
    [0.001, -0.0002, 100.0, 0.18, 0.40, 0.31, 0.72, 0.24, 0.10, 0.36, 0.63, 0.20, 0.36, 0.33]
  ]
}
```

Important: the example above shows only one cycle to keep the README short. In real usage, `window` must contain exactly 30 cycle arrays.

Required shape:

```text
window = 30 rows x 14 features
```

The 30 rows must be ordered from oldest cycle to newest cycle.

### Feature Order

Every cycle array must contain exactly these 14 values in this order:

```text
[
  op_setting_1,
  op_setting_2,
  op_setting_3,
  s2,
  s3,
  s4,
  s7,
  s8,
  s9,
  s11,
  s12,
  s13,
  s15,
  s17
]
```

Do not change this order. The ONNX model was trained with this exact column order.

### Response Body

```json
{
  "rul_cycles": 45.2,
  "alert": false,
  "confidence": 0.8712
}
```

Response fields:

- `rul_cycles`: predicted remaining useful life in cycles
- `alert`: `true` when the alert probability is at least `0.5`
- `confidence`: confidence of the predicted alert class

## Backend Integration Guide

Backend should treat the AI module as an inference microservice.

Recommended flow:

1. Collect telemetry for one engine.
2. Keep the latest 30 chronological cycles.
3. Build the 14-feature vector for each cycle using the exact feature order above.
4. Apply the same preprocessing used during training.
5. Send the resulting `30 x 14` float array to `POST /predict`.
6. Store or return the API response to the frontend.

The API validates:

- exactly 30 cycles
- exactly 14 features per cycle

If the shape is wrong, it returns HTTP `400`.

### Important Preprocessing Note

The FastAPI service does not clean raw telemetry, normalize raw sensor values, create windows, or reorder columns. It only runs inference on an already prepared model input.

Backend integration must therefore handle:

- mapping raw database fields to the required feature order
- dropping unused sensors
- applying the same scaling strategy used in training
- creating the rolling 30-cycle window
- deciding what to do when an engine has fewer than 30 cycles

For production, save and reuse the fitted preprocessing objects from training. If those scalers are not available yet, they should be exported before this model is connected to real raw telemetry.

## Frontend Integration Guide

Frontend should not send raw sensor tables directly to the model unless the frontend is responsible for preprocessing. The cleaner approach is:

```text
frontend -> main backend -> predictive maintenance API -> main backend -> frontend
```

The frontend can display:

- predicted RUL as cycles remaining
- alert state as healthy/warning/critical
- confidence as a percentage
- trend history if the backend stores previous predictions

Suggested UI logic:

```text
alert = true  -> show critical maintenance warning
alert = false -> show normal/monitoring status
```

If the frontend calls this FastAPI service directly from the browser, backend/devops may need to add FastAPI CORS middleware or expose the model API through the main backend.

## Running the API Locally

From the API directory:

```bash
cd "ai study/work/api"
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000/docs
```

## Running with Docker

Build from the API directory:

```bash
cd "ai study/work/api"
docker build -t predictive-maintenance-api .
docker run -p 8000:8000 predictive-maintenance-api
```

Deployment note: because `multitask_model.onnx` references `multitask_model.onnx.data`, the Docker image must include both files. If the Dockerfile only copies `multitask_model.onnx`, add this line:

```dockerfile
COPY multitask_model.onnx.data .
```

## Retraining or Rebuilding the Model

The training workflow is script-based:

```bash
cd "ai study/work"
python 01_eda.py
python 02_preprocessing.py
python 03_labels.py
python 04_sequences.py
python 05_model.py
python 06_train.py
python 07_evaluate.py
python 08_export_onnx.py
```

The scripts expect the raw C-MAPSS files to be available through the relative `data_dir = '../'` path. If the raw files are stored somewhere else, update `data_dir` in the scripts or place the files where the scripts expect them.

Required raw files for full retraining:

```text
train_FD001.txt
train_FD002.txt
train_FD003.txt
train_FD004.txt
test_FD001.txt
test_FD002.txt
test_FD003.txt
test_FD004.txt
RUL_FD001.txt
RUL_FD002.txt
RUL_FD003.txt
RUL_FD004.txt
```

## Current Outputs

Generated artifacts already present in the project include:

- `best_multitask_model.pth`
- `api/multitask_model.onnx`
- `api/multitask_model.onnx.data`
- preprocessed CSV files
- labeled CSV files
- learning curves
- sensor trend plots
- correlation matrix
- operating-condition cluster plot
- RUL prediction plot
- classification evaluation plot
- feature-importance plot

## Notes for Teammates

- Use the `api/` folder for deployment.
- The model input is not raw telemetry. It is a prepared 30-cycle sequence.
- The model requires exactly 14 features per cycle.
- The model requires exactly 30 cycles per prediction.
- Keep the feature order fixed.
- Keep `multitask_model.onnx` and `multitask_model.onnx.data` in the same directory.
- Add CORS or proxy through the main backend if the frontend calls the model API directly.
- Export preprocessing scalers before connecting this to real production telemetry.

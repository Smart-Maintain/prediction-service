from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np

app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time aerospace predictive maintenance API returning RUL and Failure Alert statuses."
)

# Initialize ONNX inference session
# We load the model at startup so it stays in memory and processes requests instantly
try:
    ort_session = ort.InferenceSession("multitask_model.onnx")
except Exception as e:
    print(f"Error loading ONNX model: {e}")

# Define the Request Payload Schema
class SensorReading(BaseModel):
    window: list[list[float]] = Field(
        ..., 
        description="A list of exactly 30 sequence cycles, where each sequence cycle contains exactly 14 normalized sensor features."
    )

@app.post("/predict")
def predict(data: SensorReading):
    # Input Validation: Window size
    if len(data.window) != 30:
        raise HTTPException(status_code=400, detail=f"Window size must be exactly 30. Received {len(data.window)}.")
    
    # Input Validation: Sensor counts
    for idx, cycle_data in enumerate(data.window):
        if len(cycle_data) != 14:
            raise HTTPException(status_code=400, detail=f"Each cycle must have exactly 14 features. Cycle {idx} has {len(cycle_data)}.")
    
    # Convert validated payload to numpy array shaped (1, 30, 18)
    input_data = np.array([data.window], dtype=np.float32)
    
    # Prepare ONNX Inputs
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    
    # Run Inference: (RUL string, Alert string)
    rul_out, alert_out = ort_session.run(None, ort_inputs)
    
    # Extract values from array shape (1,)
    rul_val = float(rul_out[0])
    alert_prob = float(alert_out[0])
    
    # Logic to classify the alert
    is_alert = bool(alert_prob >= 0.5)
    
    # Calculate confidence correctly mapping from Sigmoid [0-1] to % confidence of class
    confidence = alert_prob if is_alert else (1.0 - alert_prob)
    
    return {
        "rul_cycles": max(0.0, round(rul_val, 2)), # Usually cap at 0 to avoid logically defunct negative remaining cycles
        "alert": is_alert,
        "confidence": round(confidence, 4)
    }

# Run natively for testing using: uvicorn main:app --reload

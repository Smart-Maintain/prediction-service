from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
import time

app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time aerospace predictive maintenance API returning RUL and Failure Alert statuses."
)

#  JWT / API Key Authentication 
# Extract token from the Authorization Request Header
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Mock/Demo JWT Token (In real setups, this is validated via an Auth Service)
SECRET_JWT_TOKEN = "Bearer secret-token-mlops"

def validate_token(api_key: str = Security(api_key_header)):
    if api_key != SECRET_JWT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token (JWT Unauthorized)"
        )
    return api_key

#  Simple Rate Limiting (Resilience)
# In-memory storage to track request timestamps per authenticated user
user_requests = {}
RATE_LIMIT_SECONDS = 1  # Maximum 1 request allowed per second

# Initialize ONNX inference session at startup
try:
    ort_session = ort.InferenceSession("multitask_model.onnx")
except Exception as e:
    print(f"Error loading ONNX model: {e}")

class SensorReading(BaseModel):
    window: list[list[float]] = Field(
        ..., 
        description="A list of exactly 30 sequence cycles, where each sequence cycle contains exactly 14 normalized sensor features."
    )

# The endpoint is protected by adding the validate_token dependency
@app.post("/predict")
def predict(data: SensorReading, token: str = Depends(validate_token)):
    
    # Apply Rate Limiting
    current_time = time.time()
    if token in user_requests and (current_time - user_requests[token]) < RATE_LIMIT_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests! Please wait a second before trying again (Rate Limiting)."
        )
    user_requests[token] = current_time

    # --- Core Inference Logic ---
    # Input Validation: Window size checking
    if len(data.window) != 30:
        raise HTTPException(status_code=400, detail=f"Window size must be exactly 30. Received {len(data.window)}.")
    
    # Input Validation: Feature counts checking per cycle
    for idx, cycle_data in enumerate(data.window):
        if len(cycle_data) != 14:
            raise HTTPException(status_code=400, detail=f"Each cycle must have exactly 14 features. Cycle {idx} has {len(cycle_data)}.")
    
    # Convert validated payload to numpy array shaped (1, 30, 14)
    input_data = np.array([data.window], dtype=np.float32)
    
    # Prepare inputs mapping for the ONNX Runtime session
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    
    # Execute model prediction (Returns: RUL value array, Alert probability array)
    rul_out, alert_out = ort_session.run(None, ort_inputs)
    
    # Extract values from resulting arrays
    rul_val = float(rul_out[0])
    alert_prob = float(alert_out[0])
    
    # Classification logic for the binary failure alert
    is_alert = bool(alert_prob >= 0.5)
    
    # Calculate confidence mapping from Sigmoid [0-1] to percentage confidence
    confidence = alert_prob if is_alert else (1.0 - alert_prob)
    
    return {
        "rul_cycles": max(0.0, round(rul_val, 2)), # Cap at 0 to avoid logical defunct negative cycles
        "alert": is_alert,
        "confidence": round(confidence, 4)
    }
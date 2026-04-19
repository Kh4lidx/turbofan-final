from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI app
app = FastAPI(
    title="NASA Engine RUL Predictor",
    description="API for predicting Remaining Useful Life (RUL) of Turbofan Engines",
    version="1.0.0"
)

# 2. Load the Model and Scaler (The "Brain")
# We load them outside the endpoint for maximum performance
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_v1.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading assets: {e}")

# 3. Define Input Schema (Data Validation)
# This ensures the user sends EXACTLY what the model expects
class EngineFeatures(BaseModel):
    # This is a simplified version: We expect the features already engineered
    # In a real-world scenario, we would calculate rolling features inside the API
    features: list[float]

@app.get("/")
def home():
    return {"message": "NASA RUL Predictor API is Running", "docs": "/docs"}

@app.post("/predict")
def predict(data: EngineFeatures):
    try:
        # Convert input list to DataFrame for the scaler and model
        # The model expects a 2D array (1 sample, N features)
        input_data = pd.DataFrame([data.features])
        
        # 1. Scale the input using the SAME scaler from training
        scaled_data = scaler.transform(input_data)
        
        # 2. Make prediction
        prediction = model.predict(scaled_data)
        
        # 3. Return result
        return {
            "status": "success",
            "predicted_RUL": float(round(prediction[0], 2)),
            "unit": "cycles"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
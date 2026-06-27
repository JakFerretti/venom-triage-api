import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# 1. Define a flexible input schema to accommodate any number of Polars features
class ModelInput(BaseModel):
    # This expects a dictionary of features: {"Gender": 1.0, "Age": 34.0, "Local_Swelling": 0.0, ...}
    # It must strictly follow the exact column order printed by X_polars.columns in your notebook
    features: Dict[str, float] = Field(
        ..., 
        example={
        "Gender": 1.0,
        "Age": 45.0,
        "Local_Swelling": 2.0,
        "Feature_4": 0.0,
        "Feature_5": 120.0,
        "Feature_6": 0.5,
        "Feature_7": 1.0,
        "Feature_8": 0.0
  }
    )

# 2. Define the output data schema
class ModelOutput(BaseModel):
    predicted_class: str
    probability: float
    all_probabilities: Dict[str, float]

# 3. Initialize the FastAPI application
app = FastAPI(
    title="Venom Triage API",
    description="Production API serving the LightGBM Triage pipeline.",
    version="1.0.0"
)

# 4. Load the real Joblib artifacts (Matches your notebook export strategy)
PIPELINE_PATH = "venom_triage_pipeline.pkl"
ENCODER_PATH = "label_encoder_target.pkl"

try:
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Pipeline and LabelEncoder loaded successfully with Joblib!")
except Exception as e:
    pipeline = None
    label_encoder = None
    print(f"Error loading Joblib artifacts: {str(e)}")

# 5. Health Check endpoint
@app.get("/health", status_code=200)
def health_check():
    return {
        "status": "healthy" if pipeline and label_encoder else "unhealthy",
        "pipeline_loaded": pipeline is not None,
        "encoder_loaded": label_encoder is not None
    }

# 6. Prediction endpoint
@app.post("/predict", response_model=ModelOutput)
def predict(payload: ModelInput):
    if pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model components are not loaded.")
    
    try:
        # Extract values from the dictionary. 
        # CRITICAL: The JSON payload keys must follow the exact order of your training columns!
        feature_values = list(payload.features.values())
        input_data = np.array([feature_values])
        
        # 1. Generate numerical class prediction
        numeric_prediction = pipeline.predict(input_data)[0]
        
        # 2. Decode label back to original string class
        string_prediction = str(label_encoder.inverse_transform([int(numeric_prediction)])[0])
        
        # 3. Extract class probabilities
        prob_array = pipeline.predict_proba(input_data)[0]
        current_probability = float(prob_array[int(numeric_prediction)])
        
        # Map all probabilities to their respective string class names
        prob_mapping = {
            str(label_encoder.classes_[i]): float(prob_array[i]) 
            for i in range(len(label_encoder.classes_))
        }

        return ModelOutput(
            predicted_class=string_prediction, 
            probability=current_probability,
            all_probabilities=prob_mapping
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")

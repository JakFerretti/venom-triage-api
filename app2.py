import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

# =====================================================================
# 1. REQUEST & RESPONSE SCHEMAS (Pydantic Models)
# =====================================================================

class ModelInput(BaseModel):
    """
    Input schema for the clinical triage classification model.
    Contains demographic, vital, and symptoms data collected from the patient.
    """
    Age: float = Field(..., description="Age of the patient", example=30.0)
    Gender: float = Field(..., description="Gender (0 = Female, 1 = Male)", example=1.0)
    Time_Since_Bite_Min: float = Field(..., description="Time elapsed since the bite in minutes", example=120.0)
    Heart_Rate_BPM: float = Field(..., description="Heart Rate in Beats Per Minute (BPM)", example=80.0)
    Blood_Pressure_Systolic: float = Field(..., description="Systolic Blood Pressure in mm Hg", example=117.0)
    Local_Swelling: float = Field(..., description="Local Swelling Severity level (0: Mild, 1: Medium, 2: None, 3: Severe)", example=1.0)
    Muscle_Paralysis_Present: float = Field(..., description="Presence of muscle paralysis (0 = No, 1 = Yes)", example=0.0)
    Blood_Coagulation_Failure: float = Field(..., description="Presence of blood coagulation failure (0 = No, 1 = Yes)", example=0.0)

class ModelOutput(BaseModel):
    """
    Output schema containing the model's final prediction,
    the confidence score, and the complete probability distribution.
    """
    predicted_class: str = Field(..., description="The predicted class/species name")
    probability: float = Field(..., description="Confidence probability of the predicted class")
    all_probabilities: Dict[str, float] = Field(..., description="Full probability distribution across all classes")


# =====================================================================
# 2. FASTAPI APPLICATION INITIALIZATION
# =====================================================================

app = FastAPI(
    title="Venom Triage API",
    description="Production-ready API serving the LightGBM Clinical Triage pipeline.",
    version="1.0.0"
)

PIPELINE_PATH = "venom_triage_pipeline.pkl"
ENCODER_PATH = "label_encoder_target.pkl"

# Load Joblib artifacts at application startup
try:
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Inference pipeline and LabelEncoder loaded successfully!")
except Exception as e:
    pipeline = None
    label_encoder = None
    print(f"Critical error loading model artifacts: {str(e)}")


# =====================================================================
# 3. ENDPOINTS
# =====================================================================

@app.get("/health", status_code=200)
def health_check():
    """
    Simple health check endpoint to verify model readiness.
    """
    return {"status": "healthy" if pipeline and label_encoder else "unhealthy"}


@app.post("/predict", response_model=ModelOutput)
def predict(payload: ModelInput):
    """
    Endpoint for clinical triage prediction.
    Processes the payload, validates structural formats, aligns categorical mappings,
    and passes data to the Scikit-Learn Pipeline for preprocessing and LightGBM inference.
    """
    if pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model components are not loaded.")
    
    try:
        # Step 1: Extract and ensure rigorous structural/type consistency for the payload
        # Explicit type conversion protects against silent structural mismatched arrays
        raw_features = {
            'Age': float(payload.Age),
            'Gender': int(payload.Gender),
            'Time_Since_Bite_Min': float(payload.Time_Since_Bite_Min),
            'Heart_Rate_BPM': float(payload.Heart_Rate_BPM),
            'Blood_Pressure_Systolic': float(payload.Blood_Pressure_Systolic),
            'Local_Swelling': int(payload.Local_Swelling),
            'Muscle_Paralysis_Present': int(payload.Muscle_Paralysis_Present),
            'Blood_Coagulation_Failure': int(payload.Blood_Coagulation_Failure)
        }
        
        # Step 2: Construct the DataFrame preserving the exact feature sequence from training
        # The internal StandardScaler requires this rigorous alignment to prevent skewed scaling
        correct_order = [
            'Age', 'Gender', 'Time_Since_Bite_Min', 'Heart_Rate_BPM',
            'Blood_Pressure_Systolic', 'Local_Swelling', 
            'Muscle_Paralysis_Present', 'Blood_Coagulation_Failure'
        ]
        input_df = pd.DataFrame([raw_features], columns=correct_order)
        
        # Step 3: Run pipeline model inference (Scaling -> LightGBM Classification)
        numeric_prediction = pipeline.predict(input_df)[0]
        prob_array = pipeline.predict_proba(input_df)[0]
        
        # Step 4: Map target indexes back to the original string class labels
        string_prediction = str(label_encoder.inverse_transform([int(numeric_prediction)])[0])
        current_probability = float(prob_array[int(numeric_prediction)])
        
        # Generate the full distribution mapping dictionary
        prob_mapping = {
            str(label_encoder.classes_[i]): float(prob_array[i]) 
            for i in range(len(label_encoder.classes_))
        }

        # Step 5: Return validated structural output
        return ModelOutput(
            predicted_class=string_prediction, 
            probability=current_probability,
            all_probabilities=prob_mapping
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")

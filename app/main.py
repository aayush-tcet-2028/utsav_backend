from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import httpx
import shap

app = FastAPI(title="UTTSAV Smart E-Governance API", version="1.0")

# =====================================================================
# 1. LOAD PRODUCTION AI ASSETS ON STARTUP
# =====================================================================
MODEL_DIR = 'E:\\Utsav_backend\\ai_intelligence\\models\\risk_model'

try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "uttsav_preprocessor.pkl"))
    risk_model = joblib.load(os.path.join(MODEL_DIR, "uttsav_rf_model.pkl"))
    shap_explainer = joblib.load(os.path.join(MODEL_DIR, "uttsav_shap_explainer.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "uttsav_feature_names.pkl"))
    print("✅ AI Intelligence Layer Loaded Successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models. {e}")

# =====================================================================
# 2. STRICT DATA VALIDATION (PYDANTIC)
# =====================================================================
class EventApplication(BaseModel):
    Event_Category: str
    Time_Of_Day: str
    Environment_Type: str
    Expected_Crowd: int
    Max_Venue_Capacity: int
    Venue_Area_Sq_Meters: int
    Number_Of_Fire_Exits: int
    Duration_Hours: int
    Has_Fireworks: int
    Has_Temp_Structures: int
    VIP_Attendance: int
    Loudspeaker_Used: int
    Road_Closure_Required: int
    Is_Moving_Procession: int
    Food_Stalls_Present: int
    Liquor_Served: int

# =====================================================================
# 3. THE CORE INTELLIGENCE ENDPOINT
# =====================================================================
@app.post("/api/v1/applications/analyze-risk")
async def analyze_event_risk(event: EventApplication):
    try:
        # Step A: Convert Pydantic payload to DataFrame
        df_input = pd.DataFrame([event.model_dump()])
        
        # Step B: Real-time Feature Engineering (The Safety Ratios)
        df_input['Crowd_Density'] = df_input['Expected_Crowd'] / df_input['Venue_Area_Sq_Meters']
        df_input['Capacity_Utilization'] = df_input['Expected_Crowd'] / df_input['Max_Venue_Capacity']
        df_input['People_Per_Exit'] = np.where(df_input['Number_Of_Fire_Exits'] > 0, 
                                         df_input['Expected_Crowd'] / df_input['Number_Of_Fire_Exits'], 
                                         df_input['Expected_Crowd'])
        
        # Step C: Model Prediction
        X_transformed = preprocessor.transform(df_input)
        risk_prediction = risk_model.predict(X_transformed)[0]
        
        # Step D: Extract Local SHAP Logic
        shap_values = shap_explainer.shap_values(X_transformed)
        class_index = list(risk_model.classes_).index(risk_prediction)
        prediction_shap_values = shap_values[:, :, class_index][0]
        
        feature_impacts = list(zip(feature_names, prediction_shap_values))
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        
        top_3_reasons = [
            f.replace('num__', '').replace('bool__', '').replace('cat__', '') 
            for f, impact in feature_impacts[:3]
        ]
        
        # Step E: Async Call to Local Ollama LLM
        prompt = f"""
        You are an AI Safety Assistant for the UTTSAV Event Permission Portal.
        An event has been flagged as {risk_prediction.upper()} RISK.
        Event: {event.Event_Category} | Crowd: {event.Expected_Crowd}
        
        Critical factors identified by the AI Risk Engine:
        1. {top_3_reasons[0]}
        2. {top_3_reasons[1]}
        3. {top_3_reasons[2]}
        
        Write a strict, professional 2-sentence recommendation for the government official reviewing this file. 
        State the primary risk reason and suggest exactly one immediate safety precaution. Do not use conversational filler.
        """
        
        # We use httpx for non-blocking asynchronous requests
        async with httpx.AsyncClient() as client:
            try:
                # CHANGED: 'localhost' to '127.0.0.1' to fix Windows routing issues
                response = await client.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": "tinyllama", "prompt": prompt, "stream": False},
                    timeout=120.0
                )
                response.raise_for_status()
                ollama_text = response.json().get('response', '').strip()
            except Exception as e:
                # CHANGED: We are now capturing and printing the EXACT error from httpx
                ollama_text = f"API Connection Error: {str(e)} | Raw triggers: {top_3_reasons}"

        # Step F: Return Final Payload to Frontend
        return {
            "status": "success",
            "risk_level": risk_prediction,
            "ai_recommendation": ollama_text,
            "driving_factors": top_3_reasons
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import pandas as pd
import numpy as np
import joblib
import os

print("--- UTTSAV AI INFERENCE SIMULATOR ---")

# 1. Load Production Assets
MODEL_DIR = 'E:\\Utsav_backend\\ai_intelligence\\models\\risk_model'
try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "uttsav_preprocessor.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "uttsav_rf_model.pkl"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "uttsav_shap_explainer.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "uttsav_feature_names.pkl"))
    print("Successfully loaded Preprocessor, RF Model, and SHAP Explainer into memory.")
except Exception as e:
    print(f"Error loading models: {e}. Ensure you ran train_risk_model.py first.")
    exit()

# 2. Define Mock Form Submissions (Exactly as the React UI would send them)
test_payloads = [
    {
        "test_name": "Mega Political Rally (High Stampede Risk)",
        "Event_Category": "Political Rally",
        "Time_Of_Day": "Evening",
        "Environment_Type": "Outdoor",
        "Expected_Crowd": 250000,
        "Max_Venue_Capacity": 100000,          # Severe Overcrowding
        "Venue_Area_Sq_Meters": 80000,         # ~0.32 sqm per person (Lethal)
        "Number_Of_Fire_Exits": 2,             # Too few exits
        "Duration_Hours": 6,
        "Has_Fireworks": 0,
        "Has_Temp_Structures": 1,
        "VIP_Attendance": 1,
        "Loudspeaker_Used": 1,
        "Road_Closure_Required": 0,            # Traffic chaos expected
        "Is_Moving_Procession": 0,
        "Food_Stalls_Present": 1,
        "Liquor_Served": 0
    },
    {
        "test_name": "Small Private Wedding (Compliant & Low Risk)",
        "Event_Category": "Private Function",
        "Time_Of_Day": "Night",
        "Environment_Type": "Indoor",
        "Expected_Crowd": 400,
        "Max_Venue_Capacity": 1000,            # Plenty of space
        "Venue_Area_Sq_Meters": 2500,          # Massive area
        "Number_Of_Fire_Exits": 6,             # Excellent safety
        "Duration_Hours": 4,
        "Has_Fireworks": 0,
        "Has_Temp_Structures": 0,
        "VIP_Attendance": 0,
        "Loudspeaker_Used": 1,
        "Road_Closure_Required": 0,
        "Is_Moving_Procession": 0,
        "Food_Stalls_Present": 1,
        "Liquor_Served": 0
    }
]

# 3. Execution Engine
for payload in test_payloads:
    print(f"\n========================================================")
    print(f"TESTING SCENARIO: {payload['test_name']}")
    print(f"========================================================")
    
    # Remove test_name before processing
    payload_data = {k: v for k, v in payload.items() if k != 'test_name'}
    
    # Convert single JSON payload to DataFrame (required by Scikit-Learn)
    df_input = pd.DataFrame([payload_data])
    
    # --- STEP A: REAL-TIME FEATURE ENGINEERING ---
    df_input['Crowd_Density'] = df_input['Expected_Crowd'] / df_input['Venue_Area_Sq_Meters']
    df_input['Capacity_Utilization'] = df_input['Expected_Crowd'] / df_input['Max_Venue_Capacity']
    df_input['People_Per_Exit'] = np.where(df_input['Number_Of_Fire_Exits'] > 0, 
                                     df_input['Expected_Crowd'] / df_input['Number_Of_Fire_Exits'], 
                                     df_input['Expected_Crowd'])
    
    # --- STEP B: PREPROCESSING ---
    try:
        X_transformed = preprocessor.transform(df_input)
    except Exception as e:
        print(f"Preprocessing Failed: {e}")
        continue
        
    # --- STEP C: MODEL PREDICTION ---
    prediction = model.predict(X_transformed)[0]
    probabilities = model.predict_proba(X_transformed)[0]
    confidence = max(probabilities) * 100
    
    print(f">> PREDICTED RISK LEVEL:  {prediction.upper()}")
    print(f">> AI CONFIDENCE SCORE:   {confidence:.2f}%")
    
    # --- STEP D: SHAP LOCAL EXPLAINABILITY ---
    print("\n>> EXTRACTING AI LOGIC (TOP 3 DRIVING FACTORS):")
    
    # Calculate SHAP values for this specific row
    shap_values = explainer.shap_values(X_transformed)
    
    # Identify which class the model predicted (e.g., index 0 = High, 1 = Low, 2 = Medium)
    class_index = list(model.classes_).index(prediction)
    
    # Get the specific SHAP values that pushed the model toward this prediction
    prediction_shap_values = shap_values[:, :, class_index][0]
    
    # Map values to their feature names
    feature_impacts = list(zip(feature_names, prediction_shap_values))
    
    # Sort by absolute impact (highest positive impact on this specific class)
    feature_impacts.sort(key=lambda x: x[1], reverse=True)
    
    # Print the top 3 reasons
    for rank, (feature, impact) in enumerate(feature_impacts[:3], 1):
        # Format the feature name to look clean
        clean_feature = feature.replace('num__', '').replace('bool__', '').replace('cat__', '')
        print(f"   {rank}. {clean_feature} (Impact Weight: {impact:.4f})")
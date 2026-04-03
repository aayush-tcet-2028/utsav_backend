import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- UTTSAV AI PERFORMANCE EVALUATOR ---")

# 1. Load the Production Assets
MODEL_DIR = 'E:\\Utsav_backend\\ai_intelligence\\models\\risk_model'
DATA_PATH = 'E:\\Utsav_backend\\ai_intelligence\\data\\raw\\indian_event_risk_dataset_v3_balanced.csv'

try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "uttsav_preprocessor.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "uttsav_rf_model.pkl"))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# 2. Load the Full Dataset
print("\nLoading the balanced dataset (10,000+ records)...")
df = pd.read_csv(DATA_PATH)

# Separate features and target
X_raw = df.drop('Risk_Level', axis=1)
y_true = df['Risk_Level']

# 3. Apply the exact Feature Engineering used during training
print("Engineering safety ratios...")
X_raw['Crowd_Density'] = X_raw['Expected_Crowd'] / X_raw['Venue_Area_Sq_Meters']
X_raw['Capacity_Utilization'] = X_raw['Expected_Crowd'] / X_raw['Max_Venue_Capacity']
X_raw['People_Per_Exit'] = np.where(X_raw['Number_Of_Fire_Exits'] > 0, 
                                 X_raw['Expected_Crowd'] / X_raw['Number_Of_Fire_Exits'], 
                                 X_raw['Expected_Crowd'])

# 4. Process the data and Predict
print("Pushing data through the AI Pipeline...")
X_transformed = preprocessor.transform(X_raw)
y_pred = model.predict(X_transformed)

# 5. Calculate and Display the Metrics
overall_accuracy = accuracy_score(y_true, y_pred) * 100

print("\n========================================================")
print(f"OVERALL MODEL ACCURACY: {overall_accuracy:.2f}%")
print("========================================================\n")

print("DETAILED CLASSIFICATION REPORT:")
# This breaks down Precision and Recall for High, Medium, and Low risk
print(classification_report(y_true, y_pred))

print("CONFUSION MATRIX:")
# Order of labels matches the classes the model learned
labels = model.classes_
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
print(cm_df)

print("\n========================================================")
print("Evaluation Complete. This data is ready for the Hackathon Pitch Deck.")
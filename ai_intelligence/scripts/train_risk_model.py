import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Loading Balanced Indian Event Risk dataset...")
DATA_PATH = 'E:\\Utsav_backend\\ai_intelligence\\data\\raw\\indian_event_risk_dataset_v3_balanced.csv'
df = pd.read_csv(DATA_PATH)

# =====================================================================
# 1. FEATURE ENGINEERING (The Mathematical Context)
# =====================================================================
print("Calculating Safety Ratios...")
# These ratios give the AI the actual context of crowding and safety
df['Crowd_Density'] = df['Expected_Crowd'] / df['Venue_Area_Sq_Meters']
df['Capacity_Utilization'] = df['Expected_Crowd'] / df['Max_Venue_Capacity']
# Prevent division by zero for open grounds with 0 doors
df['People_Per_Exit'] = np.where(df['Number_Of_Fire_Exits'] > 0, 
                                 df['Expected_Crowd'] / df['Number_Of_Fire_Exits'], 
                                 df['Expected_Crowd'])

# =====================================================================
# 2. FEATURE SELECTION & PREPROCESSING
# =====================================================================
categorical_cols = ['Event_Category', 'Time_Of_Day', 'Environment_Type']
numeric_cols = ['Expected_Crowd', 'Max_Venue_Capacity', 'Venue_Area_Sq_Meters', 
                'Duration_Hours', 'Crowd_Density', 'Capacity_Utilization', 'People_Per_Exit']
boolean_cols = ['Has_Fireworks', 'Has_Temp_Structures', 'VIP_Attendance', 'Loudspeaker_Used',
                'Road_Closure_Required', 'Is_Moving_Procession', 'Food_Stalls_Present', 'Liquor_Served']

X = df[categorical_cols + numeric_cols + boolean_cols]
y = df['Risk_Level']

# The Preprocessor maps different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols),
        ('bool', 'passthrough', boolean_cols)
    ]
)

# =====================================================================
# 3. ENTERPRISE PIPELINE & TRAINING
# =====================================================================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

print("Initiating Grid Search and 5-Fold Cross Validation...")
# Hyperparameter grid designed for a 10k row dataset to prevent overfitting
param_grid = {
    'classifier__n_estimators': [150, 250], 
    'classifier__max_depth': [15, 25, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    rf_pipeline, 
    param_grid, 
    cv=cv_strategy, 
    scoring='accuracy', 
    n_jobs=-1 
)

# 80/20 Train-Test Split ensuring risk levels remain balanced in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model... (This may take a minute)")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nOptimization Complete. Best Parameters: {grid_search.best_params_}")

# =====================================================================
# 4. MODEL AUTHENTICATION
# =====================================================================
print("\n--- Model Authentication ---")
y_pred = best_model.predict(X_test)
print(f"Final Tuned Accuracy on Unseen Data: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# =====================================================================
# 5. XAI: EXPORTING THE LOCAL EXPLAINABILITY ENGINE
# =====================================================================
print("\n--- Generating Local Explainability (SHAP) Engine ---")

# Transform training data to build the SHAP explainer correctly
X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train)
classifier = best_model.named_steps['classifier']

# SHAP TreeExplainer maps exact mathematical logic for local explainability
explainer = shap.TreeExplainer(classifier)

# Retrieve exact feature names post-transformation for backend mapping
ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = list(ohe_feature_names) + numeric_cols + boolean_cols

# =====================================================================
# 6. EXPORTING PRODUCTION ASSETS
# =====================================================================
print("Exporting production assets to ai_intelligence/models/risk_model/...")
MODEL_DIR = 'E:\\Utsav_backend\\ai_intelligence\\models\\risk_model'
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model.named_steps['preprocessor'], os.path.join(MODEL_DIR, "uttsav_preprocessor.pkl"))
joblib.dump(classifier, os.path.join(MODEL_DIR, "uttsav_rf_model.pkl"))
joblib.dump(explainer, os.path.join(MODEL_DIR, "uttsav_shap_explainer.pkl"))
joblib.dump(all_feature_names, os.path.join(MODEL_DIR, "uttsav_feature_names.pkl"))

print("\nSuccess: Enterprise-grade Risk Model and SHAP Explainer fully trained and exported.")
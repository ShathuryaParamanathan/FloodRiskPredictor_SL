import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import shap
import os

def train_flood_model():
    # Use relative path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'sri_lanka_flood_risk_dataset_cleaned.csv')
    if not os.path.exists(data_path):
        print(f"Error: Cleaned dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Dataset Loaded. Shape:", df.shape)

    # Define Target and Features
    target = 'flood_risk_score'
    X = df.drop(columns=[target])
    X = X.astype(float) # Ensure all columns are numeric for SHAP
    y = df[target]

    # ✅ PART 2 — Data Preparation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data Split complete.")
    
    # ✅ PART 3 — Train XGBoost Model
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("Training XGBoost Model...")
    model.fit(X_train, y_train)

    # ✅ PART 4 — Model Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*30)
    print("      Model Evaluation")
    print("="*30)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print("="*30)

    # ✅ PART 5 — Add SHAP Explainability
    print("Generating SHAP Explainer (using 100 samples for speed)...")
    # Use model.predict to avoid XGBoost 2.0 conversion errors in TreeExplainer
    explainer = shap.Explainer(model.predict, shap.sample(X_train, 100))
    
    # ✅ Save Artifacts
    models_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump(model, os.path.join(models_dir, 'xgboost_flood_model.pkl'))
    joblib.dump(explainer, os.path.join(models_dir, 'shap_explainer.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'feature_columns.pkl'))
    
    default_values = X.mean().to_dict()
    joblib.dump(default_values, os.path.join(models_dir, 'default_values.pkl'))

    print(f"\nAll artifacts saved successfully in {models_dir}")

if __name__ == "__main__":
    train_flood_model()

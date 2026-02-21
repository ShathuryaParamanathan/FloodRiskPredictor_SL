import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="SL Flood Risk Predictor Dashboard", layout="wide")

# Load model and metadata
@st.cache_resource
def load_assets():
    models_dir = r'd:\Academics\L4S1\ML\Assignment\FloodRiskPredictor_SL\models'
    model = joblib.load(os.path.join(models_dir, 'xgboost_flood_model.pkl'))
    explainer = joblib.load(os.path.join(models_dir, 'shap_explainer.pkl'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    scale_cols = joblib.load(os.path.join(models_dir, 'scale_cols.pkl'))
    district_stats = joblib.load(os.path.join(models_dir, 'district_stats.pkl'))
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    return model, explainer, feature_columns, scaler, scale_cols, district_stats, label_encoders

try:
    model, explainer, feature_columns, scaler, scale_cols, district_stats, label_encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ✅ PART 3 — Prediction Function (with Internal Scaling)
def get_prediction_data(input_dict):
    # Create DataFrame from input
    input_df = pd.DataFrame([input_dict])
    
    # Apply Label Encoding for District (Model expects encoded district)
    # The stats already had encoded district as a value, but we are passing string 'district' in input_dict
    if 'district' in input_df.columns:
        input_df['district'] = label_encoders['district'].transform(input_df['district'])
    
    # Ensure all required features are present (pre-fill with zeros if any missing)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    # Apply Internal Scaling
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])
    
    # Ensure correct column order and type
    input_df = input_df[feature_columns].astype(float)
    
    # Predict
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(100, prediction))
    
    # Compute SHAP values
    shap_values = explainer(input_df)
    
    return prediction, shap_values, input_df

# ✅ PART 4 — PROFESSIONAL STREAMLIT FRONTEND
st.title("🌊 SL Flood Risk Predictor")
st.markdown("#### Predict flood risk using environmental and infrastructure indicators.")
st.markdown("---")

# ✅ PART 1 — Location-Based Interface
districts = sorted(list(district_stats.keys()))
selected_district = st.selectbox("📍 Select Your District", districts)

# ✅ District Default Logic
# Get averages for the selected district
defaults = district_stats[selected_district]

# ✅ PART 3 — REALISTIC INPUT DESIGN
st.subheader("🛠️ Adjust Parameters (Pre-filled with District Averages)")

# Create columns for grouping
col1, col2, col3 = st.columns(3)

# Initialize user_inputs with defaults for all feature columns
user_inputs = defaults.copy()
user_inputs['district'] = selected_district # Keep the string for encoding later

with col1:
    st.markdown("### 🌍 Geographic Factors")
    user_inputs['elevation_m'] = st.number_input("Elevation (meters)", value=float(defaults['elevation_m']), step=1.0)
    user_inputs['distance_to_river_m'] = st.number_input("Distance to River (meters)", value=float(defaults['distance_to_river_m']), step=10.0)

with col2:
    st.markdown("### 🌧 Rainfall Conditions")
    # Using sliders with district averages as defaults
    # We'll use reasonable ranges for SL rainfall
    user_inputs['rainfall_7d_mm'] = st.slider("Rainfall last 7 days (mm)", 0.0, 500.0, float(defaults['rainfall_7d_mm']))
    user_inputs['monthly_rainfall_mm'] = st.slider("Monthly rainfall (mm)", 0.0, 2000.0, float(defaults['monthly_rainfall_mm']))
    user_inputs['drainage_index'] = st.slider("Drainage index (0-1)", 0.0, 1.0, float(defaults['drainage_index']))

with col3:
    st.markdown("### 🏙 Urban & Population")
    user_inputs['population_density_per_km2'] = st.number_input("Population Density (per km²)", value=float(defaults['population_density_per_km2']))
    user_inputs['built_up_percent'] = st.slider("Built-up percentage (%)", 0.0, 100.0, float(defaults['built_up_percent']))

st.markdown("---")
col4, col5 = st.columns(2)

with col4:
    st.markdown("### 🏥 Infrastructure Access")
    user_inputs['infrastructure_score'] = st.slider("Infrastructure Score", 0.0, 100.0, float(defaults['infrastructure_score']))
    # Note: hospital/evac distances were not in scale_cols but they are in the dataset
    user_inputs['nearest_hospital_km'] = st.number_input("Distance to nearest hospital (km)", value=float(defaults['nearest_hospital_km']))
    user_inputs['nearest_evac_km'] = st.number_input("Distance to nearest evacuation center (km)", value=float(defaults['nearest_evac_km']))

with col5:
    st.markdown("### 🌱 Environmental Indicators (Advanced Session)")
    with st.expander("Advanced Environmental Indicators"):
        user_inputs['ndvi'] = st.slider("NDVI (Vegetation)", -1.0, 1.0, float(defaults['ndvi']))
        user_inputs['ndwi'] = st.slider("NDWI (Water Index)", -1.0, 1.0, float(defaults['ndwi']))

# Predict Button
st.markdown("---")
if st.button("🚀 Predict Flood Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing risk factors..."):
        prediction, shap_values, input_df = get_prediction_data(user_inputs)
        
        # ✅ PART 7 — RESULT DISPLAY
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("## Prediction Result")
            st.metric("Flood Risk Score", round(prediction, 2))
            
            if prediction <= 30:
                st.success("Risk Category: **Low Risk** ✅")
            elif prediction <= 60:
                st.warning("Risk Category: **Medium Risk** ⚠️")
            else:
                st.error("Risk Category: **High Risk** 🚨")
            
            st.progress(prediction / 100.0)
            
        with res_col2:
            st.markdown("## 📊 Explanation (SHAP)")
            st.write("This plot shows which features contributed most to this specific prediction.")
            
            # Local Explanation Plot (Waterfall)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)

        # Feature Impact Section
        st.markdown("### 🔍 Feature Impact Summary")
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig2)

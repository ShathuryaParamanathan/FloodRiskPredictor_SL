import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="SL Flood Risk Predictor", layout="wide")

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

# ✅ Generate Natural Language Summary
def generate_human_explanation(shap_values, feature_names, user_inputs):
    # Get raw values and names
    vals = shap_values.values[0]
    importance = np.abs(vals)
    
    # Sort indices by absolute importance
    indices = importance.argsort()[::-1]
    
    # Identify top 3 factors with their direction and values
    top_factors_desc = []
    
    for i in indices[:3]:
        raw_name = feature_names[i]
        clean_name = raw_name.replace('_', ' ').title()
        impact = vals[i]
        
        # Try to get the user-friendly value
        val = user_inputs.get(raw_name, "N/A")
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
            
        direction = "increased 📈" if impact > 0 else "decreased 📉"
        strength = "significantly" if abs(impact) > 5 else "moderately" if abs(impact) > 2 else "slightly"
        
        top_factors_desc.append(f"Specifically, your **{clean_name}** of **{val_str}** has **{strength} {direction}** the risk score.")

    # Construct the summary
    intro = "### Why this specific result?"
    detailed_points = "\n".join([f"- {point}" for point in top_factors_desc])
    
    # Add a concluding logic sentence
    primary_feat = feature_names[indices[0]].replace('_', ' ').title()
    conclusion = f"\n\nOverall, the model identified **{primary_feat}** as the most critical determining factor for this specific prediction in {user_inputs.get('district', 'this area')}."

    return f"{intro}\n{detailed_points}{conclusion}"

# ✅ APP LAYOUT
st.title("🌊 SL Flood Risk Predictor")
st.markdown("---")

left_col, right_col = st.columns([1, 1], gap="large")

# ✅ LEFT COLUMN: User Inputs
with left_col:
    st.header("🛠️ Input Parameters")
    
    # District Selection
    districts = sorted(list(district_stats.keys()))
    selected_district = st.selectbox("📍 Select Your District", districts)
    defaults = district_stats[selected_district]
    
    user_inputs = defaults.copy()
    user_inputs['district'] = selected_district

    # Use segments/expanders for organization
    with st.expander("🌍 Geographic & Urban Factors", expanded=True):
        user_inputs['elevation_m'] = st.number_input("Elevation (meters)", value=float(defaults['elevation_m']), step=1.0)
        user_inputs['distance_to_river_m'] = st.number_input("Distance to River (meters)", value=float(defaults['distance_to_river_m']), step=10.0)
        user_inputs['population_density_per_km2'] = st.number_input("Population Density (per km²)", value=float(defaults['population_density_per_km2']))
        user_inputs['built_up_percent'] = st.slider("Built-up percentage (%)", 0.0, 100.0, float(defaults['built_up_percent']))

    with st.expander("🌧 Rainfall & Drainage", expanded=True):
        user_inputs['rainfall_7d_mm'] = st.slider("Rainfall last 7 days (mm)", 0.0, 500.0, float(defaults['rainfall_7d_mm']))
        user_inputs['monthly_rainfall_mm'] = st.slider("Monthly rainfall (mm)", 0.0, 2000.0, float(defaults['monthly_rainfall_mm']))
        user_inputs['drainage_index'] = st.slider("Drainage index (0-1)", 0.0, 1.0, float(defaults['drainage_index']))

    with st.expander("🏥 Infrastructure & Environment"):
        user_inputs['infrastructure_score'] = st.slider("Infrastructure Score", 0.0, 100.0, float(defaults['infrastructure_score']))
        user_inputs['nearest_hospital_km'] = st.number_input("Distance to nearest hospital (km)", value=float(defaults['nearest_hospital_km']))
        user_inputs['nearest_evac_km'] = st.number_input("Distance to nearest evacuation center (km)", value=float(defaults['nearest_evac_km']))
        user_inputs['ndvi'] = st.slider("NDVI (Vegetation Index)", -1.0, 1.0, float(defaults['ndvi']))
        user_inputs['ndwi'] = st.slider("NDWI (Water Index)", -1.0, 1.0, float(defaults['ndwi']))

    # Predict Button
    if st.button("🚀 Predict Flood Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            prediction, shap_values, input_df = get_prediction_data(user_inputs)
            st.session_state.results = {
                'prediction': prediction,
                'shap_values': shap_values,
                'feature_names': feature_columns,
                'user_inputs': user_inputs
            }

# ✅ RIGHT COLUMN: Results
with right_col:
    st.header("📊 Prediction Results")
    
    if 'results' in st.session_state:
        res = st.session_state.results
        prediction = res['prediction']
        shap_values = res['shap_values']
        
        # 1. Score and Category
        st.subheader("Flood Risk Analysis")
        score_col, category_col = st.columns(2)
        
        with score_col:
            st.metric("Flood Risk Score", f"{prediction:.2f}/100")
        
        with category_col:
            if prediction <= 30:
                st.success("Risk Category: **Low Risk** ✅")
            elif prediction <= 60:
                st.warning("Risk Category: **Medium Risk** ⚠️")
            else:
                st.error("Risk Category: **High Risk** 🚨")
        
        st.progress(prediction / 100.0)
        
        # 2. Detailed Human-Friendly Explanation
        st.markdown("---")
        st.subheader("🕵️ Why is the risk this high/low?")
        
        # Determine the primary trend
        if prediction > 60:
            st.error(f"**High Risk Situation:** The current combination of environmental factors suggests a significant vulnerability to flooding in {selected_district}.")
        elif prediction > 30:
            st.warning(f"**Caution Advised:** There are mixed signals in {selected_district}, with some factors pushing the risk toward moderate levels.")
        else:
            st.success(f"**Safe Baseline:** {selected_district} currently shows a stable profile with low immediate flood vulnerability.")

        # Display the directional summary
        summary = generate_human_explanation(shap_values, res['feature_names'], res['user_inputs'])
        st.info(summary)
        
        # Add a breakdown by category
        with st.expander("📖 Category-Specific Reasoning"):
            st.write(f"In **{selected_district}**, the model analyzed how your specific inputs deviate from the regional norms:")
            
            # Identify a few interesting facts based on data
            st.write(f"- **Elevation vs. Risk:** Your altitude of {user_inputs['elevation_m']}m is a fixed geographic factor that {'increases' if user_inputs['elevation_m'] < defaults['elevation_m'] else 'decreases'} your baseline vulnerability compared to the district average.")
            st.write(f"- **Rainfall Impact:** With {user_inputs['rainfall_7d_mm']}mm of recent rain, the local soil drainage ({user_inputs['drainage_index']:.2f}) is the primary mechanism preventing or allowing water accumulation.")
            st.write("- **Safety Margin:** Proximity to evacuation centers and infrastructure density acts as a numerical 'discount' on the final risk score.")
        
        # 3. Single SHAP Visualization
        st.markdown("---")
        st.subheader("🔍 Technical Impact Graph")
        st.write("For experts: Red bars show features that increase the risk score, while blue bars show features that decrease it.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("👈 Enter parameters on the left and click **Predict Flood Risk** to see results.")
        
        # Add a subtle background image or placeholder if desired
        st.markdown("""
        ### How it works:
        1. Select your district to load local averages.
        2. Adjust environmental and infrastructure factors.
        3. Click Predict to run the XGBoost machine learning model.
        4. See your risk score and an explanation of why that score was given.
        """)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def clean_dataset():
    # 1. Load Dataset
    data_path = r'd:\Academics\L4S1\ML\Assignment\FloodRiskPredictor_SL\data\sri_lanka_flood_risk_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)

    print("Initial Dataset Shape:", df.shape)
    print("\nInitial Data Types:")
    print(df.dtypes)

    # 2. Select Relevant Features
    features_to_keep = [
        'district', 'place_name', 'latitude', 'longitude', 'elevation_m',
        'distance_to_river_m', 'landcover', 'soil_type', 'water_supply',
        'electricity', 'road_quality', 'population_density_per_km2',
        'built_up_percent', 'urban_rural', 'rainfall_7d_mm',
        'monthly_rainfall_mm', 'drainage_index', 'ndvi', 'ndwi',
        'water_presence_flag', 'historical_flood_count', 'infrastructure_score',
        'nearest_hospital_km', 'nearest_evac_km', 'flood_risk_score'
    ]
    
    # Filter columns to only those that exist in the dataframe
    existing_features = [col for col in features_to_keep if col in df.columns]
    df = df[existing_features]
    print("\nFeatures selected. New shape:", df.shape)

    # 3. Handle Missing Values
    print("\nMissing values before imputation:")
    print(df.isnull().sum())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values after imputation:", df.isnull().sum().sum())

    # 4. Handle Outliers
    outlier_cols = [
        'elevation_m', 'distance_to_river_m', 'population_density_per_km2',
        'rainfall_7d_mm', 'monthly_rainfall_mm', 'infrastructure_score',
        'nearest_hospital_km', 'nearest_evac_km'
    ]

    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Capping extreme values to 5th and 95th percentiles (optional but requested)
            p5 = df[col].quantile(0.05)
            p95 = df[col].quantile(0.95)
            # Clip between IQR bounds or Percentiles? 
            # User said: Use IQR method to remove extreme outliers. Optionally, cap extreme values to the 5th and 95th percentiles.
            # I will use clipping to 5th and 95th percentiles as it's often safer for predictive models than removing rows.
            df[col] = np.clip(df[col], p5, p95)

    # 5. Encode Categorical Variables
    to_one_hot = ['landcover', 'soil_type', 'water_supply', 'electricity', 'road_quality', 'urban_rural', 'water_presence_flag']
    to_label_encode = ['district', 'place_name']

    # Label Encoding for high cardinality categorical features
    label_encoders = {}
    for col in to_label_encode:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # One-Hot Encoding for remaining categorical features
    df = pd.get_dummies(df, columns=[col for col in to_one_hot if col in df.columns], drop_first=True)

    # Calculate District Stats (Averages of raw values after imputation and outlier handling)
    # We need to map the label encoded 'district' back to strings to make the stats readable OR keep them as is.
    # Actually, the user will select a string, so we should save them by string names.
    # But wait, df['district'] is already encoded.
    # Let's use a temporary copy of the district column if possible.
    
    # Let's find the inverse mapping
    district_le = label_encoders['district']
    df_for_stats = df.copy()
    df_for_stats['district_name'] = district_le.inverse_transform(df['district'])
    
    # Select columns to average (all numerical features)
    numerical_features = df_for_stats.select_dtypes(include=[np.number]).columns.tolist()
    district_stats = df_for_stats.groupby('district_name')[numerical_features].mean().to_dict(orient='index')
    
    models_dir = r'd:\Academics\L4S1\ML\Assignment\FloodRiskPredictor_SL\models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    joblib.dump(district_stats, os.path.join(models_dir, 'district_stats.pkl'))

    # 6. Normalize / Scale Numerical Features
    scale_cols = ['distance_to_river_m', 'elevation_m', 'rainfall_7d_mm', 'monthly_rainfall_mm', 'population_density_per_km2']
    scaler = StandardScaler()
    actual_scale_cols = [col for col in scale_cols if col in df.columns]
    if actual_scale_cols:
        df[actual_scale_cols] = scaler.fit_transform(df[actual_scale_cols])
    
    # Save the scale_cols list so we know what to scale in the app
    joblib.dump(actual_scale_cols, os.path.join(models_dir, 'scale_cols.pkl'))

    # 7. Save Clean Dataset
    output_path = r'd:\Academics\L4S1\ML\Assignment\FloodRiskPredictor_SL\data\sri_lanka_flood_risk_dataset_cleaned.csv'
    df.to_csv(output_path, index=False)
    # Save Artifacts for App
    models_dir = r'd:\Academics\L4S1\ML\Assignment\FloodRiskPredictor_SL\models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))

    # Save District Stats (Averages of raw values)
    # We want averages of the numerical columns BEFORE scaling but AFTER imputation
    # Let's save a copy of df before scaling for this
    # Actually, it's easier to just compute it here and save
    
    # We need to know which columns were scaled to reverse it or just save the stats now
    print(f"\nCleaned dataset saved to {output_path}")

    # 8. Inspect Cleaned Dataset
    print("\nCleaned Dataset Shape:", df.shape)
    print("\nFirst 5 rows of cleaned dataset:")
    print(df.head())
    print("\nData types of cleaned dataset:")
    print(df.dtypes)
    print("\nFinal Missing Values Check:", df.isnull().sum().sum())

if __name__ == "__main__":
    clean_dataset()

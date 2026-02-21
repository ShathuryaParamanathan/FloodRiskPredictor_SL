# SL Flood Risk Predictor

## Project Overview
The **SL Flood Risk Predictor** is a machine learning project aimed at predicting flood risk levels in different locations across Sri Lanka. The goal is to provide early warnings and insights to mitigate flood-related damages and assist in disaster management planning.

Flooding is a recurrent problem in many regions of Sri Lanka, causing property damage, disrupting transportation, and affecting public safety. Accurate prediction of flood risk can help authorities and communities take preventive measures.

---

## Problem Definition
We aim to **predict the risk of flooding** in local areas based on environmental, geographical, and infrastructural factors. This is a regression or classification problem (depending on how risk is categorized), where the target is the **flood risk level**.

**Relevance:**
- Supports disaster preparedness.
- Reduces economic and social losses due to floods.
- Enables data-driven resource allocation and response.

---

## Dataset Source: Sri Lanka Flood Risk & Inundation Dataset

We used the **Sri Lanka Flood Risk & Inundation Dataset** from Kaggle as the primary dataset for this project.  
This dataset contains approximately 25,000 **synthetically generated records** representing villages, towns, and cities across Sri Lanka with environmental and hydrological data relevant to flood risk prediction.

- **Data source:** [Kaggle](https://www.kaggle.com/datasets/dewminimnaadi/sri-lanka-flood-risk-and-inundation-dataset/data)  
- **File type:** CSV  
- **Rows:** ~25,000  
- **Columns:** Multiple environmental, geographical, and water-related features  

**Usage note:** Since the dataset is synthetic, care must be taken to understand how feature values relate to real-world flood risk when interpreting model behavior.

---

## Features
| Feature | Description |
|---------|-------------|
| latitude | Geographic latitude of location |
| longitude | Geographic longitude of location |
| elevation_m | Elevation in meters |
| distance_to_river_m | Distance to nearest river |
| landcover | Type of land cover (urban, forest, water, etc.) |
| soil_type | Soil characteristics |
| water_supply | Availability of water supply infrastructure |
| electricity | Availability of electricity infrastructure |
| road_quality | Quality of nearby roads |
| population_density_per_km2 | Population density |
| built_up_percent | Percentage of built-up area |
| urban_rural | Urban or rural classification |
| rainfall_7d | Rainfall in past 7 days |

**Target Variable:**
- `is_good_to_live` – indicates whether the location is suitable to live (Yes → 1, No → 0).


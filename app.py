import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and encoders
with open('irrigation_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('le_target.pkl', 'rb') as f:
    le_target = pickle.load(f)

st.title('Irrigation Need Predictor')
st.write('Enter field conditions to predict irrigation need')

# Numerical inputs
soil_moisture = st.slider('Soil Moisture', 0.0, 100.0, 50.0)
temperature = st.slider('Temperature (C)', 0.0, 50.0, 25.0)
rainfall = st.slider('Rainfall (mm)', 0.0, 200.0, 50.0)
wind_speed = st.slider('Wind Speed (km/h)', 0.0, 50.0, 10.0)
humidity = st.slider('Humidity', 0.0, 100.0, 60.0)
previous_irrigation = st.slider('Previous Irrigation (mm)', 0.0, 200.0, 50.0)
soil_ph = st.slider('Soil pH', 0.0, 14.0, 7.0)
organic_carbon = st.slider('Organic Carbon', 0.0, 5.0, 1.5)
electrical_conductivity = st.slider('Electrical Conductivity', 0.0, 5.0, 1.0)
sunlight_hours = st.slider('Sunlight Hours', 0.0, 16.0, 8.0)
field_area = st.slider('Field Area (hectare)', 0.0, 100.0, 10.0)

# Categorical inputs
soil_type = st.selectbox('Soil Type', ['Sandy', 'Clay', 'Loamy', 'Silt'])
crop_type = st.selectbox('Crop Type', ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Vegetables'])
growth_stage = st.selectbox('Crop Growth Stage', ['Sowing', 'Vegetative', 'Flowering', 'Harvest'])
season = st.selectbox('Season', ['Kharif', 'Rabi', 'Zaid'])
irrigation_type = st.selectbox('Irrigation Type', ['Drip', 'Sprinkler', 'Rainfed', 'Canal'])
water_source = st.selectbox('Water Source', ['Groundwater', 'Reservoir', 'River', 'Rainwater'])
mulching = st.selectbox('Mulching Used', ['Yes', 'No'])
region = st.selectbox('Region', ['North', 'South', 'East', 'West', 'Central'])



if st.button('Predict Irrigation Need'):
    # Feature engineering — same as training
    heat_wind = temperature * wind_speed
    water_balance = rainfall / (temperature + 1)
    total_water_feat = soil_moisture + rainfall
    soil_health = soil_ph * organic_carbon
    moisture_to_heat = soil_moisture / (temperature + 1)
    climate_stress = temperature * wind_speed / (humidity + 1)
    water_availability = (rainfall + soil_moisture) / (temperature + 1)
    total_water_prev = previous_irrigation + rainfall

    # Build input dataframe with exact column order as training
    input_dict = {
        'Soil_Type': [soil_type],
        'Soil_pH': [soil_ph],
        'Soil_Moisture': [soil_moisture],
        'Organic_Carbon': [organic_carbon],
        'Electrical_Conductivity': [electrical_conductivity],
        'Temperature_C': [temperature],
        'Humidity': [humidity],
        'Rainfall_mm': [rainfall],
        'Sunlight_Hours': [sunlight_hours],
        'Wind_Speed_kmh': [wind_speed],
        'Crop_Type': [crop_type],
        'Crop_Growth_Stage': [growth_stage],
        'Season': [season],
        'Irrigation_Type': [irrigation_type],
        'Water_Source': [water_source],
        'Field_Area_hectare': [field_area],
        'Mulching_Used': [mulching],
        'Previous_Irrigation_mm': [previous_irrigation],
        'Region': [region],
        'heat_wind': [heat_wind],
        'water_balance': [water_balance],
        'total_water': [total_water_feat],
        'soil_health': [soil_health],
        'Moisture_to_Heat_Ratio': [moisture_to_heat],
        'Climate_Stress_Index': [climate_stress],
        'Water_Availability_Index': [water_availability],
        'Total_Water': [total_water_prev],
        'avg_water_crop': [50.0],  # placeholder mean value
        'growth_stage_group': ['early_late' if growth_stage 
                               in ['Sowing', 'Harvesting'] else 'mid'],
        'mulching_binary': [1 if mulching == 'Yes' else 0],
        'soil_lt_25': [1 if soil_moisture < 25 else 0],
        'temp_gt_30': [1 if temperature > 30 else 0],
        'rain_lt_300': [1 if rainfall < 300 else 0],
        'wind_gt_10': [1 if wind_speed > 10 else 0],
    }

    input_df = pd.DataFrame(input_dict)
    # st.write(input_df.columns.tolist())

    # Encode categorical columns using saved encoders
    with open('le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)

    for col in le_dict:
        if col in input_df.columns:
            input_df[col] = le_dict[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)
    result = le_target.inverse_transform(prediction)[0]

    if result == 'High':
        st.error(f'Irrigation Need: HIGH')
    elif result == 'Medium':
        st.warning(f'Irrigation Need: MEDIUM')
    else:
        st.success(f'Irrigation Need: LOW')
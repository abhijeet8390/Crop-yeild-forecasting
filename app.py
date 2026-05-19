import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open("rf_model_crop.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("crop_means.pkl", "rb") as f:
    crop_means = pickle.load(f)
    
with open("le_season.pkl", "rb") as f:
    le_season = pickle.load(f)
    
with open("state_means.pkl", "rb") as f:
    state_means = pickle.load(f)
    
st.title("🌾 Indian Crop Yield Predictor")

crop = st.selectbox("Select Crop", options=list(crop_means.keys()))
season = st.selectbox("Select Season", options=["Kharif", "Rabi", "Whole Year", "Summer", "Autumn", "Winter"])
state = st.selectbox("Select State", options=list(state_means.keys()))
crop_year = st.number_input("Crop Year", min_value=1997, max_value=2030, value=2020)
area = st.number_input("Area (in hectares)", min_value=0.0, value=1000.0)
rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)

if st.button("🌱 Predict Yield"):
    crop_encoded = crop_means[crop]
    season_encoded = le_season.transform([season])[0]
    state_encoded = state_means[state]
    
    input_data = np.array([[crop_encoded, crop_year, season_encoded, state_encoded, area, rainfall]])
    
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)
    
    st.success(f"🌾 Predicted Crop Yield: {prediction:.2f} units per hectare")
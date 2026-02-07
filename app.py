import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load('house_price_model.pkl')

# 2. App Title & Description
st.title("🏡 House Price Predictor")
st.write("Enter the details of the house below to estimate its market value.")

# 3. Create Input Fields for User
# We use columns to make the layout look professional
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Total Size (SqFt)", min_value=500, max_value=10000, value=2000)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1.0, 5.0, 2.0, step=0.5)

with col2:
    age = st.number_input("Age of House (Years)", min_value=0, max_value=150, value=10)
    has_pool = st.checkbox("Has a Pool?", value=False)

# 4. Prepare Input Data for Prediction
# Convert "Has Pool" from True/False to 1/0
pool_val = 1 if has_pool else 0

# Create a DataFrame matching the EXACT columns your model expects
input_data = pd.DataFrame({
    'TotalSF': [sqft],
    'HouseAge': [age],
    'TotalBath': [bathrooms],
    'BedroomAbvGr': [bedrooms],
    'HasPool': [pool_val]
})

# 5. Make Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"Estimated Price: ${prediction:,.2f}")
    
    # Optional: Show a "confidence" range (e.g., +/- 5%)
    lower_bound = prediction * 0.95
    upper_bound = prediction * 1.05
    st.info(f"Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
import pandas as pd
import joblib
import os
import sys

# Ensure we can import from the 'src' folder
sys.path.append(os.path.abspath('src'))

from preprocessing import clean_data, feature_engineer
from model import train_model

print("1. Loading Data...")
# Load data from the data folder
df = pd.read_csv('data/train.csv')

print("2. Processing Features...")
# Apply your custom cleaning and engineering
df_clean = clean_data(df)
df_final = feature_engineer(df_clean)

# Define the exact features your App expects
features = ['TotalSF', 'HouseAge', 'TotalBath', 'BedroomAbvGr', 'HasPool']
X = df_final[features]
y = df_final['SalePrice']

print("3. Training Model...")
# Train the Random Forest
model, _, _ = train_model(X, y, model_type='random_forest')

print("4. Saving Model...")
# Save the file in the CURRENT directory
joblib.dump(model, 'house_price_model.pkl')

print(f"✅ Success! Model saved to: {os.path.abspath('house_price_model.pkl')}")
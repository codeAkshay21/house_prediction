import pandas as pd
import numpy as np

def clean_data(df):
    """
    Main function to clean raw housing data.
    """
    # 1. Handle Missing Values (Amenities)
    # For these columns, NaN means "feature not present", not "missing data"
    cols_to_fill_none = ['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu']
    for col in cols_to_fill_none:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # 2. Numerical Imputation (Fill missing numbers with Median)
    # e.g., GarageYearBuilt or LotFrontage
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def feature_engineer(df):
    """
    Creates custom features for specific project requirements:
    Size, Age, and Total Bathrooms.
    """
    df = df.copy()
    
    # 1. SIZE: Total Square Footage
    # Summing basement + 1st floor + 2nd floor
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 2. AGE: House Age at time of sale
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # 3. BATHROOMS: Total Bathrooms
    # Full bath = 1, Half bath = 0.5
    df['TotalBath'] = (df['FullBath'] + 
                       (0.5 * df['HalfBath']) + 
                       df['BsmtFullBath'] + 
                       (0.5 * df['BsmtHalfBath']))
    
    # 4. AMENITIES: Binary "Has Pool" flag
    if 'PoolQC' in df.columns:
        df['HasPool'] = df['PoolQC'].apply(lambda x: 0 if x == 'None' else 1)
        
    return df
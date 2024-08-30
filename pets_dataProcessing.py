"""
Alyssa Ang 211082M

This script contains the Data Preprocessing pipeline
1. Train, Test and Validation were derived from the raw petsTransformed.csv dataset
   - Split 70/20/10
   - Train and Target columns were also extracted
2. Performed One Hot Encoding and MinMax scaling
3. Used StratifiedShuffleSplit for splits for more consistency
"""
# Import libraries
# Import libraries
import sys 
import os

import pandas as pd
import numpy as np
import configparser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Append the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration from params.py
from config import rawData_path, processedData_path, trainCol, targetCol, testSize
from config import randomState, valSize

config = configparser.ConfigParser()
config.read('params.py')

# Initial steps of Data Preprocessing already fulfilled in EDA
# such as Feature selection and Feature Engineering

# Retrieve data, get transformed dataset
petsData = pd.read_csv(f"{rawData_path}/transformedPets.csv")

# Split into categorical and float columns for 
# OHE for categorical
# MinMax scaler for numerical
categorical_cols = ['ColorName', 'AgeBins', 'BreedBins', 'StateBins',
                    'PhotoAmtBins', 'QuantityBins','TypeName', 'GenderName', 'MaturitySizeName', 
                    'FurLengthName','HealthName', 'VaccinatedName', 'DewormedName', 'SterilizedName',
                    'BreedBinsName', 'StateBinsName']
float_cols = ['FreeorNO', 'VideoorNO', 'Quantity','BreedPure', 'ColorAmt', 'NameorNO']

# Filter existing columns
categorical_cols = [col for col in categorical_cols if col in petsData.columns]
float_cols = [col for col in float_cols if col in petsData.columns]

# One-Hot Encoding for Categorical Columns
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(petsData[categorical_cols])

# Create DataFrame with encoded features
encoded_categorical_df = pd.DataFrame(encoded_categorical, 
                                      columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate with original petsData
petsData = pd.concat([petsData, encoded_categorical_df], axis=1)
petsData.drop(columns=categorical_cols, inplace=True)

# Apply MinMax scaler to normalize numerical values
scaler = MinMaxScaler()
petsData[float_cols] = scaler.fit_transform(petsData[float_cols])

# Split into X and y
X = petsData.drop(columns=['AdoptionSpeed'])
y = petsData['AdoptionSpeed']

# Clean column names
cleaned_col_names = []
for columns in X.columns:
    cleanName = columns.replace('[','')
    cleanName = cleanName.replace(']','')
    cleanName = cleanName.replace(')','')
    cleanName = cleanName.replace('   ','')
    cleanName = cleanName.replace('  ', '')
    cleanName = cleanName.replace(' ','')
    cleaned_col_names.append(cleanName)

X.columns = cleaned_col_names

# Apply SSS to split data into test train and validation
def stratified_split(X, y, test_size=testSize, val_size=valSize, random_state=randomState):
    # First split: train + validation and test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=testSize, 
                                      random_state=randomState)
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

    # Second split: train and validation
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=valSize/(1 - testSize), 
                                     random_state=random_state)
    train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, 
                                                                  testSize, valSize,randomState)

# Print to show the size of each set
print('Shape of Train Set:', X_train.shape)
print('Shape of Validation Set:', X_val.shape)
print('Shape of Test Set:', X_test.shape)
print()
print('Length of y_train:', len(y_train))
print('Length of y_val:', len(y_val))
print('Length of y_test:', len(y_test))

# Save datasets to processedData directory
X_train.to_csv(f'{processedData_path}/X_train.csv', index=False)
X_test.to_csv(f'{processedData_path}/X_test.csv', index=False)
X_val.to_csv(f'{processedData_path}/X_val.csv', index=False)
y_train.to_csv(f'{processedData_path}/y_train.csv', 
               index=False, header=False)
y_test.to_csv(f'{processedData_path}/y_test.csv', 
              index=False, header=False)
y_val.to_csv(f'{processedData_path}/y_val.csv', 
             index=False, header=False)
"""
dataPreprocessing.py
Alyssa Ang alysappleseed@gmail.com

This script contains the data preprocessing process in the pipeline
Data preprocessing for further data transformation before model training
to ensure data quality, resulting in higher accuracy and reliability

Steps taken in this process,
1. Drop columns/features that will not contribute towards model
2. Filter by float and categorical columns for OHE and RobustScaling
> RobustScaler was used due to the distribution of this dataset being highly skewed
> RobustScaler scales features using statistics that are robust to outliers enabling good performance even in the presence of outliers or skewed distributions
3. Applied SSS for splitting
"""

# Import libraries
import sys 
import os

import pandas as pd
import numpy as np
import configparser

# Append the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration from config.py
from config import (
    enrichedData_path, targetCol, 
    testSize, randomState, valSize,
    processedData_path
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('enrichedData_path')

df = df[['customer_purchase_month',
       'customer_purchase_year', 'order_lifecycle',
       'order_estimated_lifecycle', 'order_approved_days',
       'order_to_logs_days', 'total_items', 'total_sellers_in_order',
       'product_category_name', 'distance_km', 'total_price',
       'total_freight_cost', 'product_photos_qty', 'product_name_lenght',
       'product_description_lenght', 'product_volume', 'payment_types',
       'total_installments', 'total_payment_value', 'total_sequences',
       'total_reviews', 'review_score', 'repurchasedorNO', 'volume_bins']]

# Split into categorical and float columns for 
# OHE for categorical
# Robust Scaler for numerical
categorical_cols = ['volume_bins','product_category_name', 'payment_types']

float_cols = ['customer_purchase_month',
       'customer_purchase_year', 'order_lifecycle',
       'order_estimated_lifecycle', 'order_approved_days',
       'order_to_logs_days', 'total_items', 'total_sellers_in_order',
       'distance_km', 'total_price','total_freight_cost', 
       'product_photos_qty', 'product_name_lenght',
       'product_description_lenght',
       'total_installments', 'total_payment_value', 'total_sequences',
       'total_reviews', 'review_score']

# Filter existing columns
categorical_cols = [col for col in categorical_cols if col in df.columns]
float_cols = [col for col in float_cols if col in df.columns]

# One-Hot Encoding for Categorical Columns
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(df[categorical_cols])

# Create DataFrame with encoded features
encoded_categorical_df = pd.DataFrame(encoded_categorical, 
                                      columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate with original petsData
df = pd.concat([df, encoded_categorical_df], axis=1)
df.drop(columns=categorical_cols, inplace=True)

df = df.astype('float64')

data = df[float_cols]
scaler = RobustScaler()
df[float_cols] = scaler.fit_transform(df[float_cols])

# Split into X and y
X = df.drop(columns=['repurchasedorNO'])
y = df['repurchasedorNO']

# Clean column names
cleaned_col_names = []
for columns in X.columns:
    cleanName = columns.replace('[','')
    cleanName = cleanName.replace(']','')
    cleanName = cleanName.replace(')','')
    cleanName = cleanName.replace('   ','')
    cleanName = cleanName.replace('  ', '')
    cleanName = cleanName.replace(' ','')
    cleanName = cleanName.replace('\'','')
    cleanName = cleanName.replace(',','')
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
print('Shape of Test Set:', X_test.shape)
print('Shape of Validation Set:', X_val.shape)
print()
print('Length of y_train:', len(y_train))
print('Length of y_test:', len(y_test))
print('Length of y_val:', len(y_val))
print()
print('training data:\n', y_train.value_counts())
print('\ntest data:\n', y_test.value_counts())
print('\nval data:\n', y_val.value_counts())
print("\n\nin percentages")
print('\ntraining data:\n', y_train.value_counts(normalize=True))
print('\ntest data:\n', y_test.value_counts(normalize=True))
print('\nval data:\n', y_test.value_counts(normalize=True))
print("\nrepurchasedorNO data:\n",df.repurchasedorNO.value_counts(normalize=True))

#Save datasets to processedData directory
X_train.to_csv(f'{processedData_path}/X_train.csv', index=False)
X_test.to_csv(f'{processedData_path}/X_test.csv', index=False)
X_val.to_csv(f'{processedData_path}/X_val.csv', index=False)
y_train.to_csv(f'{processedData_path}/y_train.csv', 
               index=False, header=False)
y_test.to_csv(f'{processedData_path}/y_test.csv', 
              index=False, header=False)
y_val.to_csv(f'{processedData_path}/y_val.csv', 
             index=False, header=False)

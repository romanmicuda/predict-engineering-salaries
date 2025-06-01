import pandas as pd
import torch
import joblib
import numpy as np

# Load artifacts
label_encoders = joblib.load("label_encoders.joblib")
target_encoder = joblib.load("target_encoder.joblib")
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
best_model = joblib.load("best_sklearn_wrapper.joblib")

# Load test data
test_df = pd.read_csv('data/test.csv')

# --- Preprocessing (should match train pipeline) ---
categorical_cols = ['job_title', 'job_state', 'feature_1', 'job_posted_date', 'title_state_interaction']
numerical_cols = ['feature_2', 'feature_2_times_10', 'feature_2_squared', 'feature_10_squared'] + [f'job_desc_{i:03d}' for i in range(1, 301)]
boolean_cols = ['feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 
                'feature_8', 'feature_9', 'feature_11', 'feature_12']

# Feature engineering (repeat as in train)
def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return np.nan

def extract_month(date_str):
    try:
        month_str = str(date_str)[5:]
        return int(month_str) if month_str else np.nan
    except (ValueError, TypeError):
        return np.nan

test_df['job_posted_year'] = test_df['job_posted_date'].apply(extract_year)
test_df['job_posted_month'] = test_df['job_posted_date'].apply(extract_month)

# Impute year/month if needed
from sklearn.impute import SimpleImputer
year_imputer = SimpleImputer(strategy='median')
month_imputer = SimpleImputer(strategy='median')
test_df[['job_posted_year', 'job_posted_month']] = year_imputer.fit_transform(test_df[['job_posted_year', 'job_posted_month']])

# Interaction and polynomial features
test_df['feature_2_times_10'] = test_df['feature_2'] * test_df['feature_10']
test_df['feature_2_squared'] = test_df['feature_2'] ** 2
test_df['feature_10_squared'] = test_df['feature_10'] ** 2
test_df['title_state_interaction'] = test_df['job_title'].astype(str) + '_' + test_df['job_state'].astype(str)

# Encode categorical variables
for col in categorical_cols:
    le = label_encoders[col]
    test_df[col] = le.transform(test_df[col])

# Convert boolean columns to integers
for col in boolean_cols:
    test_df[col] = test_df[col].astype(int)

# Scale numerical features
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# PCA on job_desc columns
job_desc_cols = [f'job_desc_{i:03d}' for i in range(1, 301)]
test_job_desc_pca = pca.transform(test_df[job_desc_cols])
test_pca_df = pd.DataFrame(test_job_desc_pca, columns=[f'job_desc_pca_{i+1}' for i in range(150)], index=test_df.index)
test_df = pd.concat([test_df.drop(columns=job_desc_cols), test_pca_df], axis=1)

# Prepare features for prediction
X_test = test_df.drop(columns=['obs'])

# Predict
predictions = best_model.predict(X_test)
predictions = target_encoder.inverse_transform(predictions)

# Create submission
submission = pd.DataFrame({
    'obs': test_df['obs'],
    'salary_category': predictions
})

submission.to_csv('submission_test_inference.csv', index=False)
print(submission.head())
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Define categorical, numerical, and boolean columns
categorical_cols = ['job_title', 'job_state', 'feature_1', 'job_posted_date']
numerical_cols = ['feature_2'] + [f'job_desc_{i:03d}' for i in range(1, 301)]
boolean_cols = ['feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 
                'feature_8', 'feature_9', 'feature_11', 'feature_12']

# Handle missing values
# Impute job_state with most frequent value
state_imputer = SimpleImputer(strategy='most_frequent')
train_df['job_state'] = state_imputer.fit_transform(train_df[['job_state']]).flatten()
test_df['job_state'] = state_imputer.transform(test_df[['job_state']]).flatten()

# Impute feature_10 with median
feature10_imputer = SimpleImputer(strategy='median')
train_df['feature_10'] = feature10_imputer.fit_transform(train_df[['feature_10']]).flatten()
test_df['feature_10'] = feature10_imputer.transform(test_df[['feature_10']]).flatten()

# Drop row with missing job_posted_date
train_df = train_df.dropna(subset=['job_posted_date'])

# Feature engineering: Extract year and month from job_posted_date before encoding
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

train_df['job_posted_year'] = train_df['job_posted_date'].apply(extract_year)
train_df['job_posted_month'] = train_df['job_posted_date'].apply(extract_month)
test_df['job_posted_year'] = test_df['job_posted_date'].apply(extract_year)
test_df['job_posted_month'] = test_df['job_posted_date'].apply(extract_month)

# Impute any missing year/month values with median
year_imputer = SimpleImputer(strategy='median')
month_imputer = SimpleImputer(strategy='median')
train_df[['job_posted_year', 'job_posted_month']] = year_imputer.fit_transform(train_df[['job_posted_year', 'job_posted_month']])
test_df[['job_posted_year', 'job_posted_month']] = year_imputer.transform(test_df[['job_posted_year', 'job_posted_month']])

# Additional feature engineering
# Interaction feature
train_df['feature_2_times_10'] = train_df['feature_2'] * train_df['feature_10']
test_df['feature_2_times_10'] = test_df['feature_2'] * test_df['feature_10']
numerical_cols.append('feature_2_times_10')

# Polynomial features
train_df['feature_2_squared'] = train_df['feature_2'] ** 2
test_df['feature_2_squared'] = test_df['feature_2'] ** 2
train_df['feature_10_squared'] = train_df['feature_10'] ** 2
test_df['feature_10_squared'] = test_df['feature_10'] ** 2
numerical_cols.extend(['feature_2_squared', 'feature_10_squared'])

# Categorical interaction
train_df['title_state_interaction'] = train_df['job_title'].astype(str) + '_' + train_df['job_state'].astype(str)
test_df['title_state_interaction'] = test_df['job_title'].astype(str) + '_' + test_df['job_state'].astype(str)
categorical_cols.append('title_state_interaction')

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    all_values = pd.concat([train_df[col], test_df[col]]).unique()
    le = LabelEncoder()
    le.fit(all_values)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Convert boolean columns to integers
for col in boolean_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# Scale numerical features
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Dimensionality reduction with PCA on job_desc_* columns
job_desc_cols = [f'job_desc_{i:03d}' for i in range(1, 301)]
pca = PCA(n_components=150)  # Increased components
train_job_desc_pca = pca.fit_transform(train_df[job_desc_cols])
test_job_desc_pca = pca.transform(test_df[job_desc_cols])

# Create DataFrames for PCA components
train_pca_df = pd.DataFrame(train_job_desc_pca, columns=[f'job_desc_pca_{i+1}' for i in range(150)], index=train_df.index)
test_pca_df = pd.DataFrame(test_job_desc_pca, columns=[f'job_desc_pca_{i+1}' for i in range(150)], index=test_df.index)

# Concatenate PCA components to avoid fragmentation
train_df = pd.concat([train_df.drop(columns=job_desc_cols), train_pca_df], axis=1)
test_df = pd.concat([test_df.drop(columns=job_desc_cols), test_pca_df], axis=1)

# Prepare target variable
target_encoder = LabelEncoder()
train_df['salary_category'] = target_encoder.fit_transform(train_df['salary_category'])

# Prepare features and target
X = train_df.drop(columns=['obs', 'salary_category'])
y = train_df['salary_category']
X_test = test_df.drop(columns=['obs'])

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
X_val, X_test_internal, y_val, y_test_internal = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# Custom Dataset class (unchanged)
class SalaryDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Define the neural network (unchanged)
class SalaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_classes=3, dropout_rate=0.2):
        super(SalaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.swish1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.swish2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.residual_adapter1 = nn.Linear(hidden_size, hidden_size // 2)
        
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.layer4 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.bn4 = nn.BatchNorm1d(hidden_size // 8)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.residual_adapter2 = nn.Linear(hidden_size // 4, hidden_size // 8)
        
        self.layer5 = nn.Linear(hidden_size // 8, hidden_size // 16)
        self.bn5 = nn.BatchNorm1d(hidden_size // 16)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        
        self.layer6 = nn.Linear(hidden_size // 16, num_classes)
    
    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.bn1(x1)
        x1 = self.swish1(x1)
        x1 = self.dropout1(x1)
        
        x2 = self.layer2(x1)
        x2 = self.bn2(x2)
        x2 = self.swish2(x2)
        x2 = self.dropout2(x2)
        residual1 = self.residual_adapter1(x1)
        x2 = x2 + residual1
        
        x3 = self.layer3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)
        
        x4 = self.layer4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        x4 = self.dropout4(x4)
        residual2 = self.residual_adapter2(x3)
        x4 = x4 + residual2
        
        x5 = self.layer5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)
        x5 = self.dropout5(x5)
        
        x6 = self.layer6(x5)
        return x6

# Corrected PyTorchClassifier wrapper
class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size=1024, num_classes=3, dropout_rate=0.2, 
                 learning_rate=0.00003, batch_size=128, num_epochs=50, weight_decay=1e-4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None  # Initialize classes_ attribute
    
    def fit(self, X, y):
        # Store class labels
        self.classes_ = np.unique(y)
        
        # Initialize model
        self.model = SalaryClassifier(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_classes=self.num_classes, 
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Create DataLoader
        dataset = SalaryDataset(pd.DataFrame(X), pd.Series(y))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Corrected
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            scheduler.step(epoch_loss)
        
        return self
    
    def predict(self, X):
        self.model.eval()
        predictions = []
        dataset = SalaryDataset(pd.DataFrame(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for X_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        self.model.eval()
        probabilities = []
        dataset = SalaryDataset(pd.DataFrame(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for X_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)

# Define hyperparameter grid
param_grid = {
    'hidden_size': [512, 1024],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.00003, 0.0001],
    'batch_size': [64, 128]
}

# Initialize the PyTorch classifier
base_model = PyTorchClassifier(
    input_size=X.shape[1], 
    num_classes=len(target_encoder.classes_), 
    num_epochs=50
)

# Perform grid search
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2,
    error_score='raise'
)

# Fit grid search
grid_search.fit(X_resampled, y_resampled)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Train the best model on full training data
best_model = grid_search.best_estimator_
best_model.fit(X_resampled, y_resampled)

# Evaluate on internal test set
test_predictions = best_model.predict(X_test_internal)
test_accuracy = (test_predictions == y_test_internal).mean() * 100
print(f'Internal Test Accuracy with Best Model: {test_accuracy:.2f}%')

# Predict on test set
predictions = best_model.predict(X_test)
predictions = target_encoder.inverse_transform(predictions)

# Create submission
submission = pd.DataFrame({
    'obs': test_df['obs'],
    'salary_category': predictions
})

# Save submission
submission.to_csv('submission_grid_search.csv', index=False)

# Display first few predictions
print(submission.head())
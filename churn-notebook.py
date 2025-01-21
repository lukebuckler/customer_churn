# Bank Customer Churn Prediction Analysis
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Data Loading and Initial Exploration
# Load the dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data Analysis and Visualization
# Calculate churn rate
churn_rate = (df['churn'].mean() * 100)
print(f"\nOverall churn rate: {churn_rate:.2f}%")

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='churn')
plt.title('Distribution of Churn')
plt.show()

# Analyze churn by age
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='churn', y='age')
plt.title('Age Distribution by Churn Status')
plt.show()

# Analyze churn by credit score
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='churn', y='credit_score')
plt.title('Credit Score Distribution by Churn Status')
plt.show()

# Data Preprocessing
# Create a copy of the dataframe
data = df.copy()

# Encode categorical variables
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['country'] = le.fit_transform(data['country'])

# Feature engineering
# Calculate balance per product ratio
data['balance_per_product'] = data['balance'] / (data['products_number'] + 1)

# Create age groups
data['age_group'] = pd.cut(data['age'], 
                          bins=[0, 20, 30, 40, 50, 60, 100],
                          labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+'])

# Create tenure groups
data['tenure_group'] = pd.cut(data['tenure'],
                             bins=[0, 2, 5, 8, 10],
                             labels=['0-2', '3-5', '6-8', '9-10'])

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['age_group', 'tenure_group'])

# Prepare features for modeling
# Drop unnecessary columns
features_to_drop = ['customer_id', 'churn']
X = data.drop(features_to_drop, axis=1)
y = data['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 
                     'estimated_salary', 'balance_per_product']

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Model Training
# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, 
                             max_depth=10,
                             min_samples_split=5,
                             min_samples_leaf=2,
                             random_state=42)

model.fit(X_train, y_train)

# Model Evaluation
# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.3f}")

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nCross-validation ROC-AUC Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature Importance Analysis
# Calculate and visualize feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})

# Sort by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
plt.show()

# Print top 10 most important features
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Make predictions on new data
# Example of how to make predictions on new customers
def predict_churn_probability(model, customer_data, scaler, numerical_features):
    """
    Make predictions for new customers
    """
    # Preprocess the customer data (ensure it has the same features as training data)
    customer_data[numerical_features] = scaler.transform(customer_data[numerical_features])
    
    # Make prediction
    churn_probability = model.predict_proba(customer_data)[:, 1]
    
    return churn_probability

# Example usage:
# new_customer = pd.DataFrame(...) # Create DataFrame with customer data
# churn_prob = predict_churn_probability(model, new_customer, scaler, numerical_features)

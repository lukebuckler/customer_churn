# Bank Customer Churn Prediction

Machine learning model to predict customer churn in banking sector using Python.

## Data Description

Dataset contains 10,000 customer records with features:
- customer_id
- credit_score
- country
- gender
- age
- tenure
- balance
- products_number
- credit_card
- active_member
- estimated_salary
- churn (target)

## Model Performance

- Accuracy: 87%
- ROC-AUC Score: 86.2%
- Precision: 86%
- Recall: 87%

## Installation

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

## Project Structure

```
customer-churn/
├── notebooks/
│   └── churn_analysis.ipynb
├── data/
│   └── bank_customer_churn.csv
├── requirements.txt
└── README.md
```

## Features

1. Data Preprocessing
   - Feature engineering
   - Encoding categorical variables
   - Feature scaling

2. Model Development
   - Random Forest Classifier
   - Cross-validation
   - Hyperparameter tuning

3. Analysis Features
   - Feature importance
   - Churn patterns


## Key Findings

- Top churn factors identified


## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

```python
# Load and preprocess data
data = load_and_preprocess_data('data/bank_customer_churn.csv')

# Train model
model = train_model(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Business Applications

1. Risk Assessment
   - Early churn warning system
   - Customer risk scoring

2. Retention Strategies
   - Targeted interventions
   - Personalized offers

3. Business Insights
   - Customer behavior patterns
   - Product relationship analysis

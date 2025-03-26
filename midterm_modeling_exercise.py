# -*- coding: utf-8 -*-
"""Midterm-Modeling_Exercise.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HLP-mGLohyNNtlQ6vkrNlR86YEJKjbvI
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset (replace with your dataset path or URL)
df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx")

# Data preprocessing
df = df.dropna(subset=['Customer ID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Feature engineering
customer_group = df.groupby('Customer ID').agg({
    'InvoiceDate': ['min', 'max', 'count'],
    'Quantity': 'sum',
    'Price': 'mean',
    'Invoice': 'nunique'
})

customer_group.columns = ['_'.join(col).strip() for col in customer_group.columns.values]

# Compute total revenue without deprecation warnings
df['line_total'] = df['Quantity'] * df['Price']
total_revenue = df.groupby('Customer ID')['line_total'].sum()
customer_group = customer_group.join(total_revenue.rename('total_revenue'))

# New features
customer_group['avg_order_value'] = customer_group['total_revenue'] / customer_group['Invoice_nunique']
customer_group['recency_days'] = (pd.to_datetime('2021-01-01') - customer_group['InvoiceDate_max']).dt.days

# Create binary target: repeat customer if more than one invoice
customer_group['repeat_customer'] = (customer_group['Invoice_nunique'] > 1).astype(int)

# Features and target
X = customer_group[['total_revenue', 'Quantity_sum', 'Price_mean', 'avg_order_value', 'recency_days']]
y = customer_group['repeat_customer']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
gs_rf = GridSearchCV(rf, rf_params, cv=5, scoring='f1')
gs_rf.fit(X_train, y_train)

print("Best Random Forest Params:", gs_rf.best_params_)
y_pred_rf = gs_rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, gs_rf.predict_proba(X_test)[:, 1]))

# Logistic Regression with scaling pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=10000))
])

logreg_params = {
    'logreg__C': [0.01, 0.1, 1, 10],
    'logreg__solver': ['liblinear', 'saga']
}
gs_logreg = GridSearchCV(pipeline, logreg_params, cv=5, scoring='f1')
gs_logreg.fit(X_train, y_train)

print("\nBest Logistic Regression Params:", gs_logreg.best_params_)
y_pred_lr = gs_logreg.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, gs_logreg.predict_proba(X_test)[:, 1]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
REFERENCE_DATE = pd.to_datetime('2021-01-01')
TEST_SIZE = 0.3
RANDOM_STATE = 42


def load_and_preprocess_data(url):
    """Load data and perform initial preprocessing."""
    df = pd.read_excel(url)
    df = df.dropna(subset=['Customer ID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['line_total'] = df['Quantity'] * df['Price']
    return df


def create_customer_features(df):
    """Generate customer-level features."""
    customer_group = df.groupby('Customer ID').agg({
        'InvoiceDate': ['min', 'max', 'count'],
        'Quantity': 'sum',
        'Price': 'mean',
        'Invoice': 'nunique'
    })

    customer_group.columns = ['_'.join(col).strip() for col in customer_group.columns.values]

    total_revenue = df.groupby('Customer ID')['line_total'].sum()
    customer_group = customer_group.join(total_revenue.rename('total_revenue'))

    customer_group['avg_order_value'] = customer_group['total_revenue'] / customer_group['Invoice_nunique']
    customer_group['recency_days'] = (REFERENCE_DATE - customer_group['InvoiceDate_max']).dt.days
    customer_group['repeat_customer'] = (customer_group['Invoice_nunique'] > 1).astype(int)

    return customer_group


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier with hyperparameter tuning."""
    rf_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search_rf = GridSearchCV(rf, rf_params, cv=5, scoring='f1')
    grid_search_rf.fit(X_train, y_train)
    return grid_search_rf


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model with scaling and hyperparameter tuning."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=RANDOM_STATE, max_iter=10000))
    ])

    logreg_params = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__solver': ['liblinear', 'saga']
    }
    grid_search_logreg = GridSearchCV(pipeline, logreg_params, cv=5, scoring='f1')
    grid_search_logreg.fit(X_train, y_train)
    return grid_search_logreg


def evaluate_model(model, X_test, y_test, model_name):
    """Print model evaluation metrics."""
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"{model_name} ROC-AUC:", roc_auc)


# Main script
def main():
    df = load_and_preprocess_data(DATA_URL)
    customer_features = create_customer_features(df)

    feature_columns = ['total_revenue', 'Quantity_sum', 'Price_mean', 'avg_order_value', 'recency_days']
    X = customer_features[feature_columns]
    y = customer_features['repeat_customer']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    rf_model = train_random_forest(X_train, y_train)
    print("Best Random Forest Params:", rf_model.best_params_)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    logreg_model = train_logistic_regression(X_train, y_train)
    print("\nBest Logistic Regression Params:", logreg_model.best_params_)
    evaluate_model(logreg_model, X_test, y_test, "Logistic Regression")


if __name__ == "__main__":
    main()
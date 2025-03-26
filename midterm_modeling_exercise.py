import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset from UCI repository
df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx")

# Drop rows with missing Customer ID (can't associate transactions without ID)
df = df.dropna(subset=['Customer ID'])

# Convert InvoiceDate column to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Aggregate features by Customer ID
customer_group = df.groupby('Customer ID').agg({
    'InvoiceDate': ['min', 'max', 'count'],   # First purchase, last purchase, and total invoices
    'Quantity': 'sum',                        # Total quantity purchased
    'Price': 'mean',                          # Average price of items purchased
    'Invoice': 'nunique'                      # Number of unique invoices (purchase frequency)
})

# Flatten MultiIndex column names for easier handling
customer_group.columns = ['_'.join(col).strip() for col in customer_group.columns.values]

# Compute total revenue for each customer
df['line_total'] = df['Quantity'] * df['Price']
total_revenue = df.groupby('Customer ID')['line_total'].sum()
customer_group = customer_group.join(total_revenue.rename('total_revenue'))

# Create additional engineered features
customer_group['avg_order_value'] = customer_group['total_revenue'] / customer_group['Invoice_nunique']  # AOV
customer_group['recency_days'] = (pd.to_datetime('2021-01-01') - customer_group['InvoiceDate_max']).dt.days  # Recency

# Create binary target variable: 1 if repeat customer (more than one invoice), else 0
customer_group['repeat_customer'] = (customer_group['Invoice_nunique'] > 1).astype(int)

# Select features and target variable
X = customer_group[['total_revenue', 'Quantity_sum', 'Price_mean', 'avg_order_value', 'recency_days']]
y = customer_group['repeat_customer']

# Split data into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------------------
# Random Forest Classifier with Grid Search
# -------------------------------------------

# Define hyperparameter grid
rf_params = {
    'n_estimators': [100, 200, 500],         # Number of trees
    'max_depth': [5, 10, None],              # Depth of each tree
    'min_samples_split': [2, 5, 10]          # Minimum samples required to split a node
}

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# GridSearchCV for hyperparameter tuning (5-fold CV, scoring by F1)
gs_rf = GridSearchCV(rf, rf_params, cv=5, scoring='f1')
gs_rf.fit(X_train, y_train)

# Display best parameters found
print("Best Random Forest Params:", gs_rf.best_params_)

# Predict on test set and evaluate
y_pred_rf = gs_rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, gs_rf.predict_proba(X_test)[:, 1]))

# ------------------------------------------------
# Logistic Regression Model with Pipeline & Scaling
# ------------------------------------------------

# Create a pipeline with scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),                                    # Standardize features
    ('logreg', LogisticRegression(random_state=42, max_iter=10000))  # Logistic regression with increased max_iter
])

# Hyperparameters grid for logistic regression
logreg_params = {
    'logreg__C': [0.01, 0.1, 1, 10],                # Regularization strength
    'logreg__solver': ['liblinear', 'saga']         # Different solvers for optimization
}

# GridSearchCV for logistic regression (5-fold CV, F1 scoring)
gs_logreg = GridSearchCV(pipeline, logreg_params, cv=5, scoring='f1')
gs_logreg.fit(X_train, y_train)

# Display best parameters found
print("\nBest Logistic Regression Params:", gs_logreg.best_params_)

# Predict on test set and evaluate
y_pred_lr = gs_logreg.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, gs_logreg.predict_proba(X_test)[:, 1]))

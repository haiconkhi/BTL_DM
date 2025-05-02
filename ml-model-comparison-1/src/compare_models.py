import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from models.ann import train_ann_model
from models.gradient_boosting import train_gradient_boosting_model
from models.linear_regression import train_linear_regression_model
from models.xgboost import train_xgboost_model

# Load datasets
data1 = pd.read_csv('data/Marketing_end.csv')
data2 = pd.read_csv('data/marketing_campaign.csv')

# Preprocess data (example for the first dataset)
X1 = data1.drop(columns=['Spent'])
y1 = data1['Spent']

# Split the data
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Preprocess data (example for the second dataset)
X2 = data2.drop(columns=['Response'])
y2 = data2['Response']

# Split the data
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Train and evaluate models on the first dataset
ann_model1, ann_metrics1 = train_ann_model(X_train1, y_train1)
gb_model1, gb_metrics1 = train_gradient_boosting_model(X_train1, y_train1)
lr_model1, lr_metrics1 = train_linear_regression_model(X_train1, y_train1)
xgb_model1, xgb_metrics1 = train_xgboost_model(X_train1, y_train1)

# Train and evaluate models on the second dataset
ann_model2, ann_metrics2 = train_ann_model(X_train2, y_train2)
gb_model2, gb_metrics2 = train_gradient_boosting_model(X_train2, y_train2)
lr_model2, lr_metrics2 = train_linear_regression_model(X_train2, y_train2)
xgb_model2, xgb_metrics2 = train_xgboost_model(X_train2, y_train2)

# Print performance metrics for the first dataset
print("Performance Metrics for Marketing_end.csv:")
print("ANN Metrics:", ann_metrics1)
print("Gradient Boosting Metrics:", gb_metrics1)
print("Linear Regression Metrics:", lr_metrics1)
print("XGBoost Metrics:", xgb_metrics1)

# Print performance metrics for the second dataset
print("\nPerformance Metrics for marketing_campaign.csv:")
print("ANN Metrics:", ann_metrics2)
print("Gradient Boosting Metrics:", gb_metrics2)
print("Linear Regression Metrics:", lr_metrics2)
print("XGBoost Metrics:", xgb_metrics2)
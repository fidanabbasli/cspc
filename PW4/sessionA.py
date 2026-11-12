import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# Create results directory
os.makedirs("results", exist_ok=True)

print("=== A2: CLEANING AND EXPLORATION ===")
df = pd.read_csv("data/concrete.csv")

print(f"Initial row count: {len(df)}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Row count after removing duplicates: {len(df)}")

print("\n--- Descriptive Statistics ---")
print(df.describe())

X = df.drop(columns=["strength"])
y = df["strength"]

# A3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# A4: Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

n = X_train.shape[0]
p = X_train.shape[1]
r2_train_adj = 1 - (1 - r2_train) * (n - 1) / (n - p - 1)

print("\n--- Linear Regression Results ---")
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R2: {r2_train:.4f}, Adjusted Train R2: {r2_train_adj:.4f}, Test R2: {r2_test:.4f}")

# A6 & A7: Classification (Target >= 40 MPa)
y_class = (df["strength"] >= 40).astype(int)
Xc_train, Xc_test, c_train, c_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# KNN Classifier
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(Xc_train, c_train)
c_pred = knn.predict(Xc_test)

print("\n--- KNN Classifier Metrics ---")
print(f"Accuracy: {accuracy_score(c_test, c_pred):.4f}")
print(f"Precision: {precision_score(c_test, c_pred):.4f}")
print(f"Recall: {recall_score(c_test, c_pred):.4f}")
print(f"F1 Score: {f1_score(c_test, c_pred):.4f}")

# Save metrics table for B7
metrics_df = pd.DataFrame({
    "Model": ["KNN"],
    "Accuracy": [accuracy_score(c_test, c_pred)],
    "Precision": [precision_score(c_test, c_pred)],
    "Recall": [recall_score(c_test, c_pred)],
    "F1": [f1_score(c_test, c_pred)]
})
metrics_df.to_csv("results/A_metrics.csv", index=False)
print("\nMetrics successfully saved to results/A_metrics.csv")

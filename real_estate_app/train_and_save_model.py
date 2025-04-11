import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Import your data loading and preprocessing functions
from src.data_loader import load_data
from src.preprocess import preprocess_data

# Define paths
data_path = os.path.join("data", "raw", "real_estate.csv")
model_path = os.path.join("models", "real_estate_model.joblib")

# Load and preprocess data
df_raw = load_data(data_path)            # Already returns a DataFrame
df = preprocess_data(df_raw)             # Do NOT read CSV inside preprocess

# Check if target column exists
if "price" not in df.columns:
    raise ValueError("Column 'price' not found in dataset.")

# Train-test split
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Model trained and saved at: {model_path}")

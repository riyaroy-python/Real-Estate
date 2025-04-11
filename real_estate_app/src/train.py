import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from src.utils import preprocess_data

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. MSE: {mse:.2f}")

    joblib.dump(model, 'models/real_estate_model.pkl')

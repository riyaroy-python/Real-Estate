import joblib
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def make_prediction(input_data):
    try:
        model = joblib.load('models/model.pkl')
        prediction = model.predict(np.array(input_data).reshape(1, -1))
        return prediction[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

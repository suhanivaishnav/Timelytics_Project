import joblib
import numpy as np

model = joblib.load("models/xgboost_otd_model.pkl")
test_input = np.array([[3, 7, 2023, 45000, 15000, 10, 20, 1]])

print("Predicted Delivery Time:", round(model.predict(test_input)[0], 2), "days")

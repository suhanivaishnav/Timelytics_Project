import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoders
model = joblib.load("models/xgboost_otd_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Streamlit App UI
st.title("Timelytics: Order to Delivery Time Prediction 📦")

# Sidebar - Add Image
st.sidebar.image("assets/supply_chain_optimisation.jpg", use_container_width=True)  # Add your image here
st.sidebar.markdown("### Supply Chain Optimization")

# Sidebar - User Inputs
st.sidebar.header("Enter Order Details:")
purchase_day = st.sidebar.number_input("Purchase Day", min_value=0, max_value=6, step=1)
purchase_month = st.sidebar.number_input("Purchase Month", min_value=1, max_value=12, step=1)
purchase_year = st.sidebar.number_input("Purchase Year", min_value=2016, max_value=2025, step=1)
product_size = st.sidebar.number_input("Product Size (cm³)", min_value=0.0, step=1.0)
product_weight = st.sidebar.number_input("Product Weight (g)", min_value=0.0, step=1.0)

customer_state = st.sidebar.selectbox("Customer State", label_encoders["customer_state"].classes_)
seller_state = st.sidebar.selectbox("Seller State", label_encoders["seller_state"].classes_)
shipping_method = st.sidebar.selectbox("Shipping Method", label_encoders["shipping_method"].classes_)

# Convert categorical inputs to numeric using LabelEncoders
customer_state_encoded = label_encoders["customer_state"].transform([customer_state])[0]
seller_state_encoded = label_encoders["seller_state"].transform([seller_state])[0]
shipping_method_encoded = label_encoders["shipping_method"].transform([shipping_method])[0]

# Create input DataFrame for prediction
input_data = pd.DataFrame([[purchase_day, purchase_month, purchase_year, product_size, product_weight,
                            customer_state_encoded, seller_state_encoded, shipping_method_encoded]],
                          columns=['purchase_day', 'purchase_month', 'purchase_year', 'product_size',
                                   'product_weight', 'customer_state', 'seller_state', 'shipping_method'])

# Prediction Section
st.markdown("## Predicted Delivery Time (Days)")
if st.sidebar.button("Predict Delivery Time"):
    prediction = model.predict(input_data)
    st.success(f"📦 Estimated Delivery Time: **{int(np.round(prediction[0]))} days**")

# Sample Dataset Section
st.markdown("## Sample Dataset")
sample_data = pd.DataFrame({
    "Purchase DOW": [0, 1, 2],
    "Purchase Month": [6, 3, 1],
    "Purchase Year": [2018, 2017, 2018],
    "Product Size (cm³)": [37206, 63714, 54816],
    "Product Weight (g)": [16250, 7249, 9600],
    "Customer State": [25, 25, 25],
    "Seller State": [20, 7, 20],
    "Distance (km)": [247.94, 250.35, 4.915]
})
st.dataframe(sample_data)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load datasets
df_orders = pd.read_csv('data/olist_orders_dataset.csv')
df_order_items = pd.read_csv('data/olist_order_items_dataset.csv')
df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_sellers = pd.read_csv('data/olist_sellers_dataset.csv')
df_products = pd.read_csv('data/olist_products_dataset.csv')

# Merge data
df = df_orders.merge(df_order_items, on='order_id', how='left')
df = df.merge(df_customers, on='customer_id', how='left')
df = df.merge(df_sellers, on='seller_id', how='left')
df = df.merge(df_products, on='product_id', how='left')

# Convert dates to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

# Extract new date-based features
df['purchase_day'] = df['order_purchase_timestamp'].dt.day
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_year'] = df['order_purchase_timestamp'].dt.year

# Add product size & weight (handling missing values)
df['product_size'] = df['product_length_cm'] * df['product_width_cm'] * df['product_height_cm']
df['product_size'] = df['product_size'].fillna(df['product_size'].mean())  # Fill missing product sizes
df['product_weight'] = df['product_weight_g'].fillna(df['product_weight_g'].mean())  # Fill missing values

# Add a new "shipping method" feature based on freight value
df['shipping_method'] = np.where(df['freight_value'] > df['freight_value'].median(), 'Express', 'Standard')

# Calculate delivery time (days)
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Drop missing values
df = df.dropna(subset=['delivery_days'])

# Select features
features = ['purchase_day', 'purchase_month', 'purchase_year', 'product_size', 'product_weight', 'customer_state', 'seller_state', 'shipping_method']
target = 'delivery_days'

# Encode categorical variables
label_encoders = {}
for col in ['customer_state', 'seller_state', 'shipping_method']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train-test split (Increase test size for better evaluation)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.25, random_state=42)

# Train model with better tuning
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.05, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f'Mean Absolute Error: {mae:.4f}')

# Save model
joblib.dump(model, 'models/xgboost_otd_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print('✅ Model training complete and saved.')

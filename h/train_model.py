import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load datasets
d1 = pd.read_csv('crop_price_data.csv')
d2 = pd.read_csv('Crop_Yield_Prediction.csv')
df = d1.merge(d2, on='Crop')

# Prepare features & target
X = df.drop(['Price (INR/quintal)'], axis=1)
y = df['Price (INR/quintal)']

# Convert Date column
X['Date'] = pd.to_datetime(X['Date'])
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day
X = X.drop('Date', axis=1)

# Encode categorical features
categorical_features = ['District', 'Crop', 'Market']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
X_encoded.columns = encoder.get_feature_names_out(categorical_features)

# Merge encoded features
X = X.drop(categorical_features, axis=1)
X = pd.concat([X, X_encoded], axis=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(X_train.columns, open("columns.pkl", "wb"))

print("Model & Encoder Saved!")
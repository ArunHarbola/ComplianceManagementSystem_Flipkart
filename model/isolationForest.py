import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

# Load the log dataset (replace with your dataset file)
data = pd.read_csv('./dataset/log_data2.csv')

# Convert timestamp to datetime objects
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate time difference in seconds from the minimum timestamp
min_timestamp = data['timestamp'].min()
data['time_diff'] = (data['timestamp'] - min_timestamp).dt.total_seconds()

# Select relevant features for anomaly detection
# In this case, we'll use time_diff, activity, and username
features = data[['time_diff', 'activity', 'username']]

# Convert categorical features to numerical using label encoding
label_encoders = {}
for col in ['activity', 'username']:
    label_encoders[col] = LabelEncoder()
    features[col] = label_encoders[col].fit_transform(features[col])

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Create and train the Isolation Forest model
model = IsolationForest(contamination=0.05)  # Adjust contamination based on your data
model.fit(scaled_features)

# Predict anomalies (-1) and normal instances (1)
predictions = model.predict(scaled_features)

# Add the predictions back to the original dataset
data['anomaly_prediction'] = predictions

# Print the instances predicted as anomalies
anomalies = data[data['anomaly_prediction'] == -1]
print("Anomalies:")
print(anomalies)

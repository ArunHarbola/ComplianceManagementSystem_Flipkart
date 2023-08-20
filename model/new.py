import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from sklearn.metrics import mean_squared_error

# Load the log dataset (replace with your dataset file)
data = pd.read_csv('./dataset/log_data.csv')

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

# Split the data into train and test sets
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

# ... Build and train the autoencoder model ...
input_dim = X_train.shape[1]
encoding_dim = 10  # Adjust based on your data complexity
autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(input_dim, activation='linear'))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Reconstruct the data using the trained autoencoder
reconstructed_data = autoencoder.predict(scaled_features)
mse = np.mean(np.power(scaled_features - reconstructed_data, 2), axis=1)


# Set thresholds for anomaly detection
threshold_high = np.percentile(mse, 95)
threshold_medium = np.percentile(mse, 85)

# Identify anomalies
anomalies = data[mse > threshold_high]
# Ensure 'mse' has the same length as 'anomalies'
mse = mse[:len(anomalies)]
# Assign severity levels based on thresholds
anomalies['severity'] = np.where(mse > threshold_high, 'High',
                                np.where(mse > threshold_medium, 'Medium', 'Low'))

# Define actionable insights templates
insight_templates = {
    'High': "High Severity Anomaly Detected: {details}",
    'Medium': "Medium Severity Anomaly Detected: {details}",
    'Low': "Low Severity Anomaly Detected: {details}"
}

# Generate actionable insights
actionable_insights = []
for _, anomaly_row in anomalies.iterrows():
    insight_template = insight_templates.get(anomaly_row['severity'])
    if insight_template:
        actionable_insights.append(insight_template.format(
            details=f"Timestamp: {anomaly_row['timestamp']}, User: {anomaly_row['username']}, Activity: {anomaly_row['activity']}"
        ))

# Print actionable insights
print("Actionable Insights:")
for insight in actionable_insights:
    print(insight)
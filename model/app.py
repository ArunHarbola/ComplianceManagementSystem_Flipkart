import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model('D:/MjP/Flipkart/model/autoencoder_model.h5')

# Load and preprocess the log data
log_data = pd.read_csv('./dataset/log_data.csv')
log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
min_timestamp = log_data['timestamp'].min()
log_data['time_diff'] = (log_data['timestamp'] - min_timestamp).dt.total_seconds()
log_features = log_data[['time_diff', 'activity', 'username']].values

@app.route('/')
def home():
    return "Welcome to the Anomaly Detection Service!"


@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    try:
        data = request.get_json()
        # print(data)
        input_features = data['features']

        # Perform anomaly detection using the autoencoder
        reconstructions = autoencoder.predict(np.array([input_features]))
        mse = np.mean(np.power(input_features - reconstructions, 2), axis=1)

        # Set a threshold for anomaly detection
        threshold = 0.1  # Adjust based on your data and model performance

        # Determine if the input data is an anomaly or not
        is_anomaly = mse > threshold

        return jsonify({'is_anomaly': bool(is_anomaly)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

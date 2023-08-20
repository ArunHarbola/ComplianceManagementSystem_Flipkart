
import requests
import json

url = "http://127.0.0.1:5000/detect_anomalies"
# data = {"features": [0.5, 0.7, 0.2]}
data = {"features": [10.0, 15.0, 3.0]}
# data = [
#     [10.0, 15.0, 3.0],   # Values significantly larger than usual
#     [0.1, 0.2, 0.5],     # Values significantly smaller than usual
#     [100.0, 100.0, 100.0],  # Values that are significantly uniform
#     [5.0, 2.0, 0.1],     # Values with different patterns than usual
#     [25.0, 12.0, 0.0],   # Values with unexpected zeros
#     [2.0, 10.0, 20.0],   # Values with unusual combinations
#     # Add more examples based on the characteristics of your data
# ]

headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
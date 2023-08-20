from faker import Faker
import random
import csv
import datetime

fake = Faker()

with open('big_log_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'activity', 'username', 'source_ip', 'action', 'result']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _ in range(10000):  # Generate 10,000 log entries
        writer.writerow({
            'timestamp': fake.date_time_this_year(),
            'activity': random.choice(['Login', 'ProductView', 'OrderPlaced', 'AccessDenied']),
            'username': fake.user_name(),
            'source_ip': fake.ipv4(),
            'action': random.choice(['Success', 'Failed']),
            'result': random.choice(['Success', 'Failed'])
        })
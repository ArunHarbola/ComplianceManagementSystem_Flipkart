from flask import Flask, render_template, request,redirect, url_for,Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import IsolationForest
import pickle
from keras.models import load_model
import csv
from io import StringIO
import tabula
import pandas as pd
from PyPDF2 import PdfReader  

app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')


    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')



def preprocess_data(data):
    # Convert timestamp to datetime objects
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate time difference in seconds from the minimum timestamp
    min_timestamp = data['timestamp'].min()
    data['time_diff'] = (data['timestamp'] - min_timestamp).dt.total_seconds()
    
    # Select relevant features for anomaly detection
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
    # X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)
    
    # # Build and train the autoencoder model
    # input_dim = X_train.shape[1]
    # encoding_dim = 10  # Adjust based on your data complexity
    # autoencoder = Sequential()
    # autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
    # autoencoder.add(Dense(input_dim, activation='linear'))
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    # autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

    autoencoder = load_model('./autoencoder_model.h5')
    
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
            actionable_insights.append({
                'timestamp': anomaly_row['timestamp'],
                'user': anomaly_row['username'],
                'activity': anomaly_row['activity'],
                'details': insight_template.format(
                details=f"Timestamp: {anomaly_row['timestamp']}, User: {anomaly_row['username']}, Activity: {anomaly_row['activity']}")
            })

    return scaled_features, mse, anomalies, actionable_insights
def detect_anomalies(method, data):
    if method == 'autoencoder':
        return preprocess_data(data)
    elif method == 'isolation_forest':
        return detect_anomalies_isolation_forest(data)
    else:
        raise ValueError("Invalid method")

def detect_anomalies_isolation_forest(data):
    # Convert timestamp to datetime objects
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate time difference in seconds from the minimum timestamp
    min_timestamp = data['timestamp'].min()
    data['time_diff'] = (data['timestamp'] - min_timestamp).dt.total_seconds()
    
    # Select relevant features for anomaly detection
    features = data[['time_diff', 'activity', 'username']]
    
    # Convert categorical features to numerical using label encoding
    label_encoders = {}
    for col in ['activity', 'username']:
        label_encoders[col] = LabelEncoder()
        features[col] = label_encoders[col].fit_transform(features[col])
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Initialize and fit the Isolation Forest model
    isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination based on your data
    isolation_forest.fit(scaled_features)
    
    # Predict anomaly scores
    anomaly_scores = isolation_forest.decision_function(scaled_features)
    
    # Set thresholds for anomaly detection
    threshold = np.percentile(anomaly_scores, 5)  # Example: consider bottom 5% as anomalies
    
    # Identify anomalies
    anomalies = data[anomaly_scores < threshold]
    
    # Define actionable insights templates
    insight_templates = {
        'Anomaly': "Anomaly Detected: {details}"
    }
    
    # Generate actionable insights
    actionable_insights = []
    for _, anomaly_row in anomalies.iterrows():
        actionable_insights.append({
                'timestamp': anomaly_row['timestamp'],
                'user': anomaly_row['username'],
                'activity': anomaly_row['activity'],
                'details': f"Anomaly Detected: Timestamp: {anomaly_row['timestamp']}, User: {anomaly_row['username']}, Activity: {anomaly_row['activity']}"
            })
    return scaled_features, anomaly_scores, anomalies, actionable_insights 


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename.endswith('.pdf'):
                 # Convert the PDF to a DataFrame
                df = tabula.read_pdf(uploaded_file, pages='all')
    
                # Save the DataFrame to a CSV file
                df.to_csv('output.csv')
            elif uploaded_file.filename.endswith(('.csv', '.txt' )):
                    # Use the uploaded file as is
                data = uploaded_file
            else:
                raise ValueError("Unsupported file format")
            data = pd.read_csv(uploaded_file)
            method = request.form.get('method')

            scaled_features, mse, anomalies, actionable_insights = detect_anomalies(method, data)

            # Save the data for later use
            with open('model_data.pkl', 'wb') as f:
                pickle.dump((scaled_features, mse, anomalies, actionable_insights), f)

            # Redirect to the 'results' route
            return redirect(url_for('results'))

    return render_template('index.html')


@app.route('/Login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/SignUp', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    
    return render_template('signup.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/results', methods=['GET'])
def results():
    # Load the saved data
    with open('model_data.pkl', 'rb') as f:
        scaled_features, mse, anomalies, actionable_insights = pickle.load(f)

    # Render the 'results.html' template with the actionable insights
    return render_template('results.html', insights=actionable_insights)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive File IDs (Replace with real IDs for all files)
MODEL_FILES = {
    "knn.joblib": "1ueZ8rG20z0H0p8bqKJz4WjIsn0yHN394",
    "le_Region.joblib": "1ueZ8rG20z0H0p8bqKJz4WjIsn0yHN394",
    "le_Soil_Type.joblib": "1ueZ8rG20z0H0p8bqKJz4WjIsn0yHN394",
    "le_Crop.joblib": "1ueZ8rG20z0H0p8bqKJz4WjIsn0yHN394",
    "le_Weather_Condition.joblib": "1ueZ8rG20z0H0p8bqKJz4WjIsn0yHN394",
    "minmax_scaler.joblib": "1ueZ8rG20z0H0p8bqKJz4WjIsn0yHN394"
}

# Download missing models
for filename, file_id in MODEL_FILES.items():
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filepath, quiet=False)

# Load models
model = joblib.load(os.path.join(MODEL_DIR, "knn.joblib"))
le_region = joblib.load(os.path.join(MODEL_DIR, "le_Region.joblib"))
le_soil = joblib.load(os.path.join(MODEL_DIR, "le_Soil_Type.joblib"))
le_crop = joblib.load(os.path.join(MODEL_DIR, "le_Crop.joblib"))
le_weather = joblib.load(os.path.join(MODEL_DIR, "le_Weather_Condition.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "minmax_scaler.joblib"))

# Landing Page
@app.route('/')
def landing():
    return render_template('landing.html')

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Region': request.form['Region'],
            'Soil_Type': request.form['Soil_Type'],
            'Crop': request.form['Crop'],
            'Rainfall_mm': float(request.form['Rainfall_mm']),
            'Temperature_Celsius': float(request.form['Temperature_Celsius']),
            'Fertilizer_Used': request.form['Fertilizer_Used'],
            'Irrigation_Used': request.form['Irrigation_Used'],
            'Weather_Condition': request.form['Weather_Condition']
        }

        X = preprocess_input(user_input)
        scaled_pred = model.predict(X)[0]
        final_yield = scaler.inverse_transform([[0, 0, scaled_pred]])[0][2]

        return render_template('index.html', prediction=round(final_yield, 2))

    return render_template('index.html', prediction=None)

# Preprocessing logic
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    df['Region'] = le_region.transform(df['Region'])
    df['Soil_Type'] = le_soil.transform(df['Soil_Type'])
    df['Crop'] = le_crop.transform(df['Crop'])
    df['Weather_Condition'] = le_weather.transform(df['Weather_Condition'])

    df['Irrigation_Used'] = 1 if df['Irrigation_Used'].iloc[0] == 'Yes' else 0
    df['Fertilizer_Used'] = 1 if df['Fertilizer_Used'].iloc[0] == 'Yes' else 0

    scaled_values = scaler.transform([
        [df['Rainfall_mm'].iloc[0], df['Temperature_Celsius'].iloc[0], 0]
    ])[0]

    df['Rainfall_mm'] = scaled_values[0]
    df['Temperature_Celsius'] = scaled_values[1]

    return df[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm',
               'Temperature_Celsius', 'Fertilizer_Used',
               'Irrigation_Used', 'Weather_Condition']]

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)

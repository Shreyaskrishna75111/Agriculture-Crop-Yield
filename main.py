from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import gdown
import zipfile

app = Flask(__name__)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_ZIP_PATH = os.path.join(os.path.dirname(__file__), 'models.zip')
GDRIVE_FILE_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE'  # Replace with your actual file ID
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'

# Ensure models directory exists
if not os.path.exists(MODEL_DIR):
    print("Downloading model zip from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_ZIP_PATH, quiet=False)
    print("Extracting model files...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    os.remove(MODEL_ZIP_PATH)

# Load the trained model and preprocessing tools
model = joblib.load(os.path.join(MODEL_DIR, "knn.joblib"))
le_region = joblib.load(os.path.join(MODEL_DIR, "le_Region.joblib"))
le_soil = joblib.load(os.path.join(MODEL_DIR, "le_Soil_Type.joblib"))
le_crop = joblib.load(os.path.join(MODEL_DIR, "le_Crop.joblib"))
le_weather = joblib.load(os.path.join(MODEL_DIR, "le_Weather_Condition.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "minmax_scaler.joblib"))

@app.route('/')
def landing():
    return render_template('landing.html')

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

if __name__ == '__main__':
    app.run(debug=True)

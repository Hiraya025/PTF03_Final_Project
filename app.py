from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Fixed paths for Vercel deployment
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model', 'best_model.pkl')
scaler_path = os.path.join(base_dir, 'model', 'scaler.pkl')
le_path = os.path.join(base_dir, 'model', 'label_encoder.pkl')

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Try loading the label encoder if it exists
if os.path.exists(le_path):
    label_encoder = joblib.load(le_path)
else:
    label_encoder = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get ONLY the 8 specific inputs (User does NOT input AQI anymore)
        co = float(request.form['co'])
        no = float(request.form['no'])
        no2 = float(request.form['no2'])
        o3 = float(request.form['o3'])
        so2 = float(request.form['so2'])
        pm2_5 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        nh3 = float(request.form['nh3'])

        # 2. Prepare and scale input array
        features = np.array([[co, no, no2, o3, so2, pm2_5, pm10, nh3]])
        features_scaled = scaler.transform(features)

        # 3. Predict the AQI
        raw_prediction = model.predict(features_scaled)[0]

        # Decode if using a classifier, otherwise keep the numeric value
        if label_encoder is not None:
            try:
                predicted_aqi = label_encoder.inverse_transform([int(raw_prediction)])[0]
            except Exception:
                predicted_aqi = raw_prediction
        else:
            predicted_aqi = raw_prediction

        # 4. Interpret the AQI Risk Level
        try:
            aqi_val = float(predicted_aqi)
            predicted_aqi = round(aqi_val, 2)
            if aqi_val <= 50:
                interpretation = "Low Risk (Good)"
            elif aqi_val <= 100:
                interpretation = "Moderate Risk"
            elif aqi_val <= 150:
                interpretation = "Unhealthy for Sensitive Groups"
            elif aqi_val <= 200:
                interpretation = "High Risk (Unhealthy)"
            else:
                interpretation = "Very High Risk (Hazardous)"
        except ValueError:
            # Fallback if the model predicts a string category directly
            interpretation = str(predicted_aqi)

        return render_template('result.html',
                               aqi=predicted_aqi,
                               interpretation=interpretation,
                               co=co, no=no, no2=no2, o3=o3,
                               so2=so2, pm25=pm2_5, pm10=pm10, nh3=nh3)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

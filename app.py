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

# Load all three models
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(le_path)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get AQI and the 8 specific inputs
        aqi = float(request.form['aqi'])
        co = float(request.form['co'])
        no = float(request.form['no'])
        no2 = float(request.form['no2'])
        o3 = float(request.form['o3'])
        so2 = float(request.form['so2'])
        pm2_5 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        nh3 = float(request.form['nh3'])

        # Prepare input array (Only the 8 gases/particulates for the scaler)
        features = np.array([[co, no, no2, o3, so2, pm2_5, pm10, nh3]])

        # Scale inputs
        features_scaled = scaler.transform(features)

        # Predict numeric value, then decode it to text using the label encoder
        numeric_prediction = model.predict(features_scaled)
        risk_label = label_encoder.inverse_transform(numeric_prediction)[0]

        return render_template('result.html',
                               aqi=aqi,
                               prediction=risk_label,
                               co=co, no=no, no2=no2, o3=o3,
                               so2=so2, pm25=pm2_5, pm10=pm10, nh3=nh3)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

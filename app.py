from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model', 'best_model.pkl')
scaler_path = os.path.join(base_dir, 'model', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Risk label mapping (adjust based on what your ML Modeler used)
RISK_LABELS = {0: "Low Risk", 1: "High Risk"}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        aqi = float(request.form['aqi'])
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])

        # Prepare input array — order must match training data!
        features = np.array([[temperature, humidity, aqi, pm25, pm10]])

        # Scale inputs
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        risk_label = RISK_LABELS.get(int(prediction), "Unknown")

        return render_template('result.html',
                               prediction=risk_label,
                               temperature=temperature,
                               humidity=humidity,
                               aqi=aqi,
                               pm25=pm25,
                               pm10=pm10)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

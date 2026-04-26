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

# Risk label mapping
RISK_LABELS = {0: "Low Risk", 1: "High Risk"}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the 8 specific inputs required by the scaler
        co = float(request.form['co'])
        no = float(request.form['no'])
        no2 = float(request.form['no2'])
        o3 = float(request.form['o3'])
        so2 = float(request.form['so2'])
        pm2_5 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        nh3 = float(request.form['nh3'])

        # Prepare input array — order matches training data components: 
        # co, no, no2, o3, so2, pm2_5, pm10, nh3
        features = np.array([[co, no, no2, o3, so2, pm2_5, pm10, nh3]])

        # Scale inputs
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        risk_label = RISK_LABELS.get(int(prediction), "Unknown")

        return render_template('result.html',
                               prediction=risk_label,
                               co=co, no=no, no2=no2, o3=o3,
                               so2=so2, pm25=pm2_5, pm10=pm10, nh3=nh3)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

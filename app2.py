from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# load model
model = joblib.load("predictive_maintenance_model.pkl")

# simple HTML page
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Predictive Maintenance Demo</title>
</head>
<body>
    <h2>Engine Failure Prediction</h2>
    <form method="post" action="/predict_ui">
        <label>Enter Sensor Values (example numbers):</label><br><br>

        sensor1: <input type="text" name="sensor1"><br><br>
        sensor2: <input type="text" name="sensor2"><br><br>
        sensor3: <input type="text" name="sensor3"><br><br>
        sensor4: <input type="text" name="sensor4"><br><br>

        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <h3>Failure Probability: {{prediction}}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(html_page)

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    data = {
        "sensor1": float(request.form["sensor1"]),
        "sensor2": float(request.form["sensor2"]),
        "sensor3": float(request.form["sensor3"]),
        "sensor4": float(request.form["sensor4"])
    }

    df = pd.DataFrame([data])
    prediction = model.predict_proba(df)[0][1]

    return render_template_string(html_page, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
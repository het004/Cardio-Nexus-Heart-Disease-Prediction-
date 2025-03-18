from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask App
app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("venv/random_forest_model.pkl", "rb"))
scaler = pickle.load(open("venv/standard_scaler.pkl", "rb"))


# Route for index page
@app.route('/')
def index():
    return render_template("index.html")

# Route for home page with form
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get user input
            input_data = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]

            # Scale the input data
            input_data_scaled = scaler.transform([input_data])

            # Predict using the model
            prediction = model.predict(input_data_scaled)

            # Interpret the result
            result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"

            return render_template("home.html", prediction=result)

        except Exception as e:
            return render_template("home.html", prediction=f"Error: {e}")

    return render_template("home.html", prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('BMI-Prediction.pkl')


@app.route('/')
def home():
    return render_template('index_bmi.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # # sex = float(request.form['Sex'])
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        df = np.array([[age, height, weight]])
        print(df)
        out = model.predict(df)
        print(out)
        return render_template('index_bmi.html', prediction_text=f'Your BMI count is {round(out[0], 2)}')
    else:
        return render_template('index_bmi.html')


if __name__ == '__main__':
    app.run(debug=True)

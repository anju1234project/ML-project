from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features = features.split(',')


    if len(features) != model.n_features_in_:
        return render_template('index.html', message=[
            f"⚠️ Invalid input: Expected {model.n_features_in_} features, got {len(features)}."])

    try:
        np_features = np.asarray(features, dtype=np.float32)
    except ValueError:
        return render_template('index.html', message=["⚠️ Invalid input: All values must be numbers."])

    pred = model.predict(np_features.reshape(1, -1))
    message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']
    return render_template('index.html', message=message)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)

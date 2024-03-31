import pickle
from flask import Flask,request, app, jsonify ,url_for, render_template
import numpy as np
import pandas as pd


# Create flask app
app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

# Create app route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])


if __name__== "__main__":
    app.run(debug=True)



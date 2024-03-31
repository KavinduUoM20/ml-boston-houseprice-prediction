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
    print(output)
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    try:
        if request.method == 'POST' and request.form:
            data = [float(x) for x in request.form.values()]
            
            if data:
                final_input = scaler.transform(np.array(data).reshape(1, -1))
                print(final_input)
                output = regmodel.predict(final_input)[0]
                return render_template('home.html', predicted_val=output)
            else:
                return "Error: No data submitted."
        else:
            return "Error: No form data received."
    except Exception as e:
        # Handle exceptions gracefully
        return f"An error occurred: {str(e)}", 500

if __name__== "__main__":
    app.run(debug=True)



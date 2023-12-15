import pickle
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

model = load('xgb_model.bin')

app = Flask('capstone')

@app.route('/predict', methods=['POST'])
def predict():
    material_components = request.get_json()
    
    df = pd.DataFrame([material_components])
    y_pred = model.predict(df)
    
    result = {
        'proof_stress_mpa': round(float(y_pred[0][0]), 3),
        'tensile_strength_mpa': round(float(y_pred[0][1]), 3),
        'elongation_perc': round(float(y_pred[0][2]), 3),
        'reduction_in_area_perc': round(float(y_pred[0][3]), 3)
    } 

    return jsonify(result)

from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('Best_model_power_prediction.joblib')
    features = joblib.load('Features.joblib')
    print('Models and features list successfully loaded')
except Exception as e:
    print(f'Error while loading model or feature list :- {e}')
    model = None
    features = None



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if model is None or features is None:
        return jsonify({'Error' : 'model or Features is None'}) , 500

    try:
       data = request.get_json(force=True)
       if not data:
           return jsonify({'Error' : 'Data not provided'}) , 400
       
        # Create a dictionary for the input features, providing default values
       input_data = {
            'AMBIENT_TEMPERATURE': float(data.get('AMBIENT_TEMPERATURE', 0)),
            'MODULE_TEMPERATURE': float(data.get('MODULE_TEMPERATURE', 0)),
            'IRRADIATION': float(data.get('IRRADIATION', 0)),
            'IRRADIATION_LAG_1': float(data.get('IRRADIATION_LAG_1', 0)),
            'hour': int(data.get('hour', 0)),
            'day_of_week': int(data.get('day_of_week', 0)),
            'month': int(data.get('month', 0)),
        }
        
       
       input_df = pd.DataFrame([data])

       final_features = list(features)
       for feature in input_data.keys():
           if feature not in final_features:
               final_features.append(feature)
       
       input_df = input_df.reindex(columns=final_features,fill_value=0)
       
       
        
       print('Received Data :-', data)

       prediction = model.predict(input_df)[0]
       
       response = {
           'prediction' : float(prediction)
       }

    
       return jsonify(response)
    
    except Exception as e:
        print('Error occured while predicting', e )
        return jsonify({'error' : str(e)}) , 500


if __name__ == '__main__':
    app.run(debug=True)




import pickle

from flask import Flask
from flask import request
from flask import jsonify

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()          # assume input as json parse and turn into python dictionary

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5                  # decide threshold so mkt not need to do that

    result = {                             # prepare output as json
        'churn_probability': float(y_pred),
        'churn': bool(churn)

    }

    return jsonify(result)

if __name__ == "__main__":             # python main method
    app.run(debug=True, host='0.0.0.0', port=9696)

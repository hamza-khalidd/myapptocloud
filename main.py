import pickle
from flask import Flask, request, jsonify


app = Flask('predictions')

@app.route('/', methods=['POST'])
def predict():
    vehicle = request.get_json()
    with open('./model_files/model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
        vv = vehicle
    predictions = model.predict_proba([vv])
    pp = predictions.tolist()
    result = {
        'predictions': pp
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)


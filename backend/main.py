from flask import Flask, request, jsonify
from model import main, make_prediction
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    print(data)
    # pre-process data
    input_data = [data['A1_Score'], data['A2_Score'], data['A3_Score'], data['A4_Score'], data['A5_Score'], data['A6_Score'], data['A7_Score'], data['A8_Score'], data['A9_Score'], data['A10_Score'], data['autism'], data['result'], data['ethnicity']]
    print(input_data)
    # make prediction
    prediction = make_prediction(input_data)
    prediction = round(prediction*100, 2)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(port=5000, debug=True)

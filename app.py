# app.py — local testing only
from flask import Flask, request, jsonify, send_from_directory
from predict_logic import predict_survival

app = Flask(__name__, static_folder=".")

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = predict_survival(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=3000)
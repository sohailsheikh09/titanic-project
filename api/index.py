# api/index.py
from flask import Flask, request, jsonify, send_from_directory
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from predict_logic import predict_survival

app = Flask(__name__, static_folder="..", static_url_path="")

@app.route("/")
def home():
    return send_from_directory("..", "index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = predict_survival(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
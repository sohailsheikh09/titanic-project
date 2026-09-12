# predict_logic.py
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

def predict_survival(data):
    input_df = pd.DataFrame([{
        "Pclass": data["pclass"],
        "Sex": data["sex"],
        "Age": data["age"],
        "SibSp": data["sibsp"],
        "Parch": data["parch"],
        "Fare": data["fare"],
        "Embarked": data["embarked"],
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "survived": int(prediction),
        "probability": round(float(probability), 3)
    }
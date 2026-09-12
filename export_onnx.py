# export_onnx.py
import joblib
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType, FloatTensorType

model = joblib.load("model.pkl")

# Define input types matching your feature columns and their dtypes
# Order must match the column order used in training: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
initial_types = [
    ("Pclass", FloatTensorType([None, 1])),
    ("Sex", StringTensorType([None, 1])),
    ("Age", FloatTensorType([None, 1])),
    ("SibSp", FloatTensorType([None, 1])),
    ("Parch", FloatTensorType([None, 1])),
    ("Fare", FloatTensorType([None, 1])),
    ("Embarked", StringTensorType([None, 1])),
]

onnx_model = to_onnx(
    model,
    initial_types=initial_types,
    options={id(model): {"zipmap": False}}
)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Exported model.onnx successfully")
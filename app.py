from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

feature_columns = joblib.load("feature_columns.pkl")
scaler = joblib.load("scaler.pkl")

class SalesRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = SalesRegressor(input_dim=len(feature_columns))
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([{
            "Store": data["Store"],
            "Holiday_Flag": data["Holiday_Flag"],
            "Temperature": data["Temperature"],
            "Fuel_Price": data["Fuel_Price"],
            "CPI": data["CPI"],
            "Unemployment": data["Unemployment"],
            "Year": data["Year"],
            "Month": data["Month"],
            "Week": data["Week"]
        }])

        input_data = input_data[feature_columns]
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        return jsonify({
            "predicted_weekly_sales": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
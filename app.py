from flask import Flask, render_template, request
import numpy as np
import pickle
import shap
import pandas as pd

app = Flask(__name__)

# Load model + columns
model = pickle.load(open("model/price_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# SHAP explainer
explainer = shap.Explainer(model)


# ---------- FEATURE BUILDER ----------
def build_features(form_data):
    data = {}

    data["room_size"] = float(form_data["room_size"])
    data["furnishing"] = int(form_data["furnishing"])
    data["AC"] = int(form_data["AC"])
    data["availability_365"] = float(form_data["availability"])
    data["number_of_reviews"] = float(form_data["reviews"])
    data["reviews_per_month"] = float(form_data["rpm"])

    # Derived features
    data["demand_score"] = data["number_of_reviews"] * 0.5 + data["reviews_per_month"] * 10
    data["amenities_score"] = data["availability_365"]/36 + data["reviews_per_month"]*2

    data["base_price"] = 50 + data["room_size"]*2 + data["furnishing"]*20
    data["demand_factor"] = 1 + (data["demand_score"] / 100)
    data["supply_factor"] = 1 - (data["availability_365"]/365)*0.3
    data["quality_factor"] = 1 + data["amenities_score"]*0.05 + data["AC"]*0.1 + data["furnishing"]*0.1

    # Initialize all columns
    final_input = dict.fromkeys(columns, 0)

    for key in data:
        if key in final_input:
            final_input[key] = data[key]

    # Location encoding
    loc = form_data["location"]

    if "neighbourhood_group_Manhattan" in final_input:
        final_input["neighbourhood_group_Manhattan"] = 1 if loc == "Manhattan" else 0

    if "neighbourhood_group_Brooklyn" in final_input:
        final_input["neighbourhood_group_Brooklyn"] = 1 if loc == "Brooklyn" else 0

    return pd.DataFrame([final_input])


# ---------- SHAP FUNCTION (NOT A ROUTE) ----------
def get_shap_explanation(input_df):
    shap_values = explainer(input_df)

    values = shap_values.values[0]
    features = input_df.columns

    top_idx = np.argsort(np.abs(values))[-3:][::-1]

    results = []
    for i in top_idx:
        direction = "⬆️" if values[i] > 0 else "⬇️"
        results.append(f"{features[i]} {direction} ({values[i]:.2f})")

    return results


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features_df = build_features(request.form)

        pred = model.predict(features_df)[0]

        shap_results = get_shap_explanation(features_df)

        surge = "⚡ Surge Pricing" if pred > 300 else "Normal Pricing"

        return render_template(
            "index.html",
            prediction_text=f"{pred:.2f}",
            surge_text=surge,
            shap_results=shap_results
        )

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
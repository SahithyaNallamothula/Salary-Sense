
from flask import Flask, render_template, request
import numpy as np
import pickle
from pathlib import Path

app = Flask(__name__)

# Load the trained machine learning model
MODEL_PATH = Path(__file__).resolve().parent / "SLR_MODEL.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/")
def fun():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def prediction():
    try:
        # Get experience from the HTML form
        experience = request.form.get("experience", "").strip()

        if not experience:
            return render_template(
                "index.html",
                error="Please enter your years of experience."
            )

        years = float(experience)

        if not np.isfinite(years) or not 0 <= years <= 60:
            return render_template(
                "index.html",
                error="Enter experience between 0 and 60 years."
            )

        # Model input must be two-dimensional
        features = np.array([[years]])

        # Predict salary
        result = float(model.predict(features)[0])

        return render_template(
            "index.html",
            prediction_text=f"{result:,.2f}"
        )

    except (TypeError, ValueError):
        return render_template(
            "index.html",
            error="Please enter a valid number."
        )


if __name__ == "__main__":
    app.run(debug=True)

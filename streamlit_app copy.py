import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


st.set_page_config(layout="wide")
st.title("Saponification Twin")

st.markdown(
    """
    How to use:
    1) Post initial NaOH addition, note amount added and pH
    2) Enter numeric values for each feature below.
    3) Click Predict to see the model's output.
    """
)


# Expected model filename (placed alongside this app file)
MODEL_NAME = "LR_model_2.pkl"
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / MODEL_NAME


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}\nPlease place {MODEL_NAME} next to this app file.")
    with path.open("rb") as f:
        return pickle.load(f)


st.sidebar.header("Model status")
try:
    model = load_model(MODEL_PATH)
    st.sidebar.success(f"Loaded model: {MODEL_NAME}")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# If the model was saved as a dict with metadata, accept that too
if isinstance(model, dict) and "model" in model:
    saved = model
    model = saved.get("model")
    features = saved.get("features")
else:
    saved = None
    # Hard-coded feature list based on model_gen.py
    features = [
        "Batch Size",
        "InitialAddition",
        "pH",
        "Temp_1",
        "pH.1",
        "Temp_2",
        "JHM",
        "R55",
    ]


st.markdown("---")
st.header("Make a single prediction")

default_values = {f: 0.0 for f in features}
input_vals = {}
cols = st.columns(max(1, min(4, len(features))))
for i, feat in enumerate(features):
    col = cols[i % len(cols)]
    input_vals[feat] = col.number_input(feat, value=float(default_values[feat]))

predict_btn = st.button("Predict")
if predict_btn:
    try:
        x_arr = np.array([list(input_vals.values())]).astype(float)
        pred = model.predict(x_arr)
        # handle various shapes
        try:
            pred_val = float(np.array(pred).ravel()[0])
        except Exception:
            pred_val = float(pred)
        st.success(f"Predicted value: {pred_val:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


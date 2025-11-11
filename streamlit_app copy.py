import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ast


st.set_page_config(layout="wide")
st.title("Saponification pH Correction Twin")

st.markdown(
    """
    
    How to use:
    1) Post initial NaOH addition, note amount added and pH
    2) Enter numeric values for each feature below.
    3) pH Required for JHM: 8.5, R55:8.9 
    4) Click Predict to see the model's output.


    """
)


# Expected model filename (placed alongside this app file)
MODEL_NAME = "LR_model_3.pkl"
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

def load_features_from_model_gen(base_dir: Path):
    """Parse model_gen.py in base_dir and extract a top-level `cols` list if present."""
    try:
        mg = base_dir / "model_gen.py"
        if not mg.exists():
            return None
        src = mg.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "cols":
                        try:
                            val = ast.literal_eval(node.value)
                            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                                return val
                        except Exception:
                            return None
        return None
    except Exception:
        return None


# If the model was saved as a dict with metadata, accept that too
if isinstance(model, dict) and "model" in model:
    saved = model
    model = saved.get("model")
    features = saved.get("features")
else:
    saved = None
    # Prefer authoritative feature list from model_gen.py when possible
    features = load_features_from_model_gen(BASE)
    if not features:
        # fallback hard-coded feature list
        features = [
            "Batch Size (MT)",
            "Initial Addition (kg)",
            "pH_Initial",
            "Temp_1",
            "pH_Required",
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
    if feat in ["JHM", "R55"]:
        # Radio buttons for binary features
        val = col.radio(
            f"{feat} (Yes/No)", 
            options=["No", "Yes"],
            horizontal=True,
            key=f"radio_{feat}"
        )
        # Convert Yes/No to 1/0 for the model
        input_vals[feat] = 1.0 if val == "Yes" else 0.0
    else:
        # Regular number input for other features
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
        st.success(f"Predicted value (kg): {pred_val:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


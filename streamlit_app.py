import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
import pickle
import io
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Saponification Twin")

st.markdown(
    """
    Upload a CSV or use your dataframe to train a the model.
    After training, enter values for the selected features and click Predict.
    """
)

# Sidebar: upload and options
with st.sidebar:
    st.header("Data / Model Controls")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"]) 
    model_file = st.file_uploader("Upload pretrained model (pickle, optional)", type=["pkl", "pickle"]) 
    scaler_file = st.file_uploader("Upload pretrained scaler/preprocessor (pickle, optional)", type=["pkl", "pickle"]) 
    test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state (int)", value=42, step=1)
    model_type = st.selectbox("Model type", options=["Linear Regression", "Gaussian Process"], index=0)
    # Gaussian process settings (shown only when selected for training)
    if model_type == "Gaussian Process":
        st.markdown("**Gaussian Process kernel settings**")
        gp_kernel_choice = st.selectbox("Base kernel", options=["RBF", "Matern"], index=0)
        gp_length_scale = st.number_input("Length scale", value=1.0, step=0.1, format="%.3f")
        gp_constant = st.number_input("Constant kernel multiplier", value=1.0, step=0.1, format="%.3f")
        gp_noise = st.number_input("WhiteKernel noise level", value=1e-1, format="%.5f")
    train_button = st.button("Train Model")

# Load dataframe
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("No CSV uploaded — you may instead upload a pretrained model (see sidebar) or upload a CSV to train from data.")
    df = None

# Load pretrained model if provided
pretrained_model = None
pretrained_features = None
model_source_description = None
pretrained_model_type = None
pretrained_scaler = None
if 'model_file' in locals() and model_file is not None:
    try:
        model_file.seek(0)
        loaded = pickle.load(model_file)
        # Common pattern: saved as dict with model and features
        if isinstance(loaded, dict) and "model" in loaded:
            pretrained_model = loaded.get("model")
            pretrained_features = loaded.get("features")
            # optional model_type saved when exporting
            pretrained_model_type = loaded.get("model_type")
            # optional scaler saved when exporting
            pretrained_scaler = loaded.get("scaler")
            model_source_description = "pickle (dict: model + features)"
        else:
            # If raw model saved, accept it and ask user for feature names
            pretrained_model = loaded
            # try to infer model type from class name
            try:
                pretrained_model_type = getattr(pretrained_model, "__class__").__name__
            except Exception:
                pretrained_model_type = None
            model_source_description = "pickle (raw model)"
    except Exception as e:
        st.sidebar.error(f"Failed loading pretrained model: {e}")
        pretrained_model = None
# If user uploaded a separate scaler file, load it (overrides scaler in model if present)
if 'scaler_file' in locals() and scaler_file is not None:
    try:
        scaler_file.seek(0)
        pretrained_scaler = pickle.load(scaler_file)
        st.sidebar.write("Loaded pretrained scaler/preprocessor from file")
    except Exception as e:
        st.sidebar.error(f"Failed loading pretrained scaler: {e}")

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Select numeric columns only for regression
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the uploaded data. Linear regression requires numeric features and a numeric target.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.write("Select features and target")
    features = st.sidebar.multiselect("Feature columns (X)", options=numeric_cols, default=numeric_cols[:-1])
    target = st.sidebar.selectbox("Target column (y)", options=numeric_cols, index=len(numeric_cols)-1)

    if target in features:
        st.sidebar.warning("Target column removed from features automatically.")
        features = [c for c in features if c != target]

    if not features:
        st.warning("Please select at least one feature column.")

    # Storage for trained model and data
    model = None
    X_train = X_test = y_train = y_test = None
    trained = False

    if train_button:
        if not features:
            st.error("No feature columns selected.")
        else:
            X = df[features].values
            y = df[[target]].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )
            # Choose model type for training
            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                X_test_used = X_test
            else:
                # Build kernel
                if gp_kernel_choice == "RBF":
                    base = RBF(length_scale=gp_length_scale)
                else:
                    base = Matern(length_scale=gp_length_scale)
                kernel = ConstantKernel(constant_value=gp_constant) * base + WhiteKernel(noise_level=gp_noise)
                model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                # Fit a scaler and apply to X for GPR
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # GPR expects 1d y
                model.fit(X_train_scaled, y_train.ravel())
                X_test_used = X_test_scaled
            trained = True

            # Metrics
            y_pred = model.predict(X_test_used)
            # normalize shapes to 1d for metrics
            y_true_flat = np.array(y_test).ravel()
            y_pred_flat = np.array(y_pred).ravel()
            try:
                mse = mean_squared_error(y_true_flat, y_pred_flat)
                r2 = r2_score(y_true_flat, y_pred_flat)
            except Exception:
                mse = float('nan')
                r2 = float('nan')

            st.success("Model trained")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model parameters")
                if model_type == "Linear Regression" and hasattr(model, "coef_"):
                    coefs = pd.DataFrame({"feature": features, "coefficient": model.coef_.flatten()})
                    st.dataframe(coefs)
                    # intercept may be scalar or array
                    try:
                        intercept_val = float(model.intercept_[0])
                    except Exception:
                        intercept_val = float(np.array(model.intercept_).ravel()[0])
                    st.write("Intercept:", intercept_val)
                else:
                    st.write("Model type:", model_type)
                    st.write("Kernel:", getattr(model, "kernel_", getattr(model, "kernel", "(not available)")))

            with col2:
                st.subheader("Evaluation")
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("R-squared", f"{r2:.4f}")

            # Plots
            st.subheader("Diagnostics")
            if len(features) == 1:
                # scatter + line
                fig, ax = plt.subplots()
                ax.scatter(X_test.flatten(), y_test.flatten(), label="test")
                sorted_idx = np.argsort(X_test.flatten())
                ax.plot(X_test.flatten()[sorted_idx], y_pred_flat[sorted_idx], color="red", label="pred")
                ax.set_xlabel(features[0])
                ax.set_ylabel(target)
                ax.legend()
                st.pyplot(fig)
            else:
                # predicted vs actual
                fig, ax = plt.subplots()
                ax.scatter(y_true_flat, y_pred_flat)
                ax.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], "k--", lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

            # Allow model download (include scaler for GPR if present)
            buffer = io.BytesIO()
            export_dict = {"model": model, "features": features, "model_type": model_type}
            if model_type == "Gaussian Process":
                try:
                    export_dict["scaler"] = scaler
                except Exception:
                    pass
            pickle.dump(export_dict, buffer)
            buffer.seek(0)
            st.download_button("Download trained model (pickle)", data=buffer, file_name="trained_model.pkl")

    # If model trained in this session, show input boxes for making a single prediction
    st.markdown("---")
    st.header("Make a single prediction")
    if features:
        if df is not None:
            default_values = {f: float(df[f].mean()) for f in features}
        else:
            default_values = {f: 0.0 for f in features}

        input_vals = {}
        cols = st.columns(max(1, min(4, len(features))))
        for i, feat in enumerate(features):
            col = cols[i % len(cols)]
            input_vals[feat] = col.number_input(feat, value=default_values[feat])

        predict_btn = st.button("Predict")
        if predict_btn:
            # If not trained in this session, train on full dataset automatically
            if not trained:
                st.info("Model not trained in this session — training on full dataset using selected features.")
                X = df[features].values
                y = df[[target]].values
                if model_type == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X, y)
                else:
                    if gp_kernel_choice == "RBF":
                        base = RBF(length_scale=gp_length_scale)
                    else:
                        base = Matern(length_scale=gp_length_scale)
                    kernel = ConstantKernel(constant_value=gp_constant) * base + WhiteKernel(noise_level=gp_noise)
                    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                    # Fit scaler for full-data training
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y.ravel())

            x_arr = np.array([list(input_vals.values())]).astype(float)
            # If model is GPR and scaler is available, scale the inputs
            try:
                if isinstance(model, GaussianProcessRegressor) or (pretrained_model_type and "GaussianProcess" in str(pretrained_model_type)):
                    # prefer scaler from pretrained_scaler (if using pretrained model), otherwise use local scaler
                    scaler_to_use = None
                    if 'pretrained_scaler' in locals() and pretrained_scaler is not None:
                        scaler_to_use = pretrained_scaler
                    elif 'scaler' in locals() and scaler is not None:
                        scaler_to_use = scaler
                    if scaler_to_use is not None:
                        x_arr = scaler_to_use.transform(x_arr)
            except Exception:
                pass

            pred = model.predict(x_arr)
            pred_val = float(np.array(pred).ravel()[0])
            st.success(f"Predicted {target}: {pred_val:.4f}")

            # Show contribution (coefficient * value) if available
            if hasattr(model, "coef_"):
                try:
                    coefs = np.array(model.coef_).flatten()
                    contrib = pd.DataFrame({"feature": features, "value": list(input_vals.values()), "coef": coefs})
                    contrib["contribution"] = contrib["value"] * contrib["coef"]
                    st.subheader("Feature contributions")
                    st.dataframe(contrib)
                except Exception:
                    st.info("Could not compute feature contributions for this model type.")
            else:
                st.info("Feature contributions are not available for this model type.")

elif pretrained_model is not None:
    # Provide a prediction UI using the uploaded pretrained model (no CSV required)
    st.subheader("Pretrained model prediction")
    st.write(model_source_description)

    # If the pretrained model didn't include feature names, ask the user to provide them
    if pretrained_features is None:
        # Try several automatic inference strategies before asking the user:
        def infer_feature_names(model):
            # 1) sklearn estimator attribute
            try:
                if hasattr(model, "feature_names_in_"):
                    vals = getattr(model, "feature_names_in_")
                    # normalize to python strings
                    return [str(v) for v in list(vals)]
            except Exception:
                pass

            # 2) estimator or pipeline exposing get_feature_names_out
            try:
                if hasattr(model, "get_feature_names_out"):
                    names = model.get_feature_names_out()
                    if len(names) > 0:
                        return [str(n) for n in list(names)]
            except Exception:
                pass

            # 3) If it's a Pipeline, try to call get_feature_names_out on the pipeline
            try:
                from sklearn.pipeline import Pipeline

                if isinstance(model, Pipeline):
                    try:
                        names = model.get_feature_names_out()
                        if len(names) > 0:
                            return list(names)
                    except Exception:
                        # try extracting from the transformer part
                        try:
                            transformer = model[:-1]
                            names = transformer.get_feature_names_out()
                            if len(names) > 0:
                                return list(names)
                        except Exception:
                            pass
            except Exception:
                pass

            # 4) ColumnTransformer or fitted transformers may expose get_feature_names_out
            try:
                from sklearn.compose import ColumnTransformer

                if isinstance(model, ColumnTransformer):
                    try:
                        names = model.get_feature_names_out()
                        if len(names) > 0:
                            return list(names)
                    except Exception:
                        pass
            except Exception:
                pass

            return None

        inferred = infer_feature_names(pretrained_model)
        if inferred:
            pretrained_features = inferred
            st.write("Inferred feature names from model:")
            st.write(pretrained_features)
        else:
            st.info("Could not infer feature names automatically from the uploaded model.")
            st.write("Options: upload a small training CSV to infer column names automatically, or (as last resort) enter names manually.")
            sample_file = st.file_uploader("Upload small CSV used for training (optional)", type=["csv"], key="feat_csv")
            if sample_file is not None:
                try:
                    sample_df = pd.read_csv(sample_file)
                    # prefer numeric columns if possible
                    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        pretrained_features = numeric_cols
                    else:
                        pretrained_features = sample_df.columns.tolist()
                    st.success("Extracted feature names from uploaded CSV")
                    st.write(pretrained_features)
                except Exception as e:
                    st.error(f"Failed to read sample CSV: {e}")
            else:
                # As an additional automatic fallback, if the model exposes n_features_in_ we can generate placeholder names
                try:
                    if hasattr(pretrained_model, "n_features_in_"):
                        n = int(getattr(pretrained_model, "n_features_in_"))
                        pretrained_features = [f"feature_{i}" for i in range(n)]
                        st.write("Auto-generated placeholder feature names from model.n_features_in_:")
                        st.write(pretrained_features)
                    else:
                        feat_input = st.text_input("Enter feature names (comma-separated) for the pretrained model (manual fallback)")
                        if feat_input:
                            pretrained_features = [f.strip() for f in feat_input.split(",") if f.strip()]
                except Exception:
                    feat_input = st.text_input("Enter feature names (comma-separated) for the pretrained model (manual fallback)")
                    if feat_input:
                        pretrained_features = [f.strip() for f in feat_input.split(",") if f.strip()]

    if pretrained_features:
        features = pretrained_features
        default_values = {f: 0.0 for f in features}

        input_vals = {}
        cols = st.columns(max(1, min(4, len(features))))
        for i, feat in enumerate(features):
            col = cols[i % len(cols)]
            input_vals[feat] = col.number_input(feat, value=default_values[feat])

        predict_btn = st.button("Predict (pretrained model)")
        if predict_btn:
            model = pretrained_model
            if model is None:
                st.error("Uploaded pretrained model could not be loaded.")
            else:
                x_arr = np.array([list(input_vals.values())]).astype(float)
                try:
                    # If pretrained model is a GPR and a scaler was provided, apply scaling
                    try:
                        is_gpr = isinstance(model, GaussianProcessRegressor) or (pretrained_model_type and "GaussianProcess" in str(pretrained_model_type))
                    except Exception:
                        is_gpr = False
                    if is_gpr and pretrained_scaler is not None:
                        try:
                            x_arr = pretrained_scaler.transform(x_arr)
                        except Exception:
                            # fallback: try reshape
                            x_arr = np.array(x_arr).astype(float)
                    pred = model.predict(x_arr)
                    val = float(np.array(pred).ravel()[0])
                    st.success(f"Predicted: {val:.4f}")

                    # If possible, show contributions
                    if hasattr(model, "coef_"):
                        coefs = np.array(model.coef_).flatten()
                        contrib = pd.DataFrame({"feature": features, "value": list(input_vals.values()), "coef": coefs})
                        contrib["contribution"] = contrib["value"] * contrib["coef"]
                        st.subheader("Feature contributions")
                        st.dataframe(contrib)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.info("Please provide feature names for the pretrained model (comma-separated) in the input above.")

else:
    st.stop()

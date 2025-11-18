import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------
# Load artifact (comes entirely from training script)
# -----------------------------------------------------------
artifact = joblib.load("meningioma_models.joblib")

model_A  = artifact["model_A"]
model_B  = artifact["model_B"]
model_AB = artifact["model_AB"]

feature_names = artifact["feature_names"]

# calibration bins
bins_A  = artifact["reliability_A"]
bins_B  = artifact["reliability_B"]
bins_AB = artifact["reliability_AB"]

# validation metrics
val_AtoB = artifact["validation_AtoB"]
val_BtoA = artifact["validation_BtoA"]
val_AB   = artifact["validation_AB"]


# -----------------------------------------------------------
# UI / Input helpers
# -----------------------------------------------------------
location_map = {
    0: "infratentorial",
    1: "supratentorial",
    2: "skullbase",
    3: "convexity",
}

def make_row(age, size, loc_code, epilepsy, ich, focal, calc, edema):
    row = {
        "age": age,
        "tumorsize": size,
        "epilepsi": int(epilepsy),
        "tryksympt": int(ich),
        "focalsympt": int(focal),
        "calcified": int(calc),
        "edema": int(edema),
    }
    df = pd.DataFrame([row])

    # zero location dummies
    for c in feature_names:
        if c.startswith("location_"):
            df[c] = 0

    chosen = f"location_{location_map[loc_code]}"
    if chosen in feature_names:
        df[chosen] = 1

    # ensure all columns present
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0

    return df[feature_names]


def lookup_bin(p, bins):
    for b in bins:
        if b["p_min"] <= p <= b["p_max"]:
            return b
    return None


# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="Meningioma Risk", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")

age = st.sidebar.number_input("Age", 18, 100, 65)
size = st.sidebar.number_input("Tumor size (mm)", 1, 150, 30)
loc = st.sidebar.selectbox(
    "Location", list(location_map.keys()),
    format_func=lambda x: location_map[x].capitalize()
)

yn = {0: "No", 1: "Yes"}
ep = st.sidebar.selectbox("Epilepsy", [0,1], format_func=lambda x: yn[x])
ich = st.sidebar.selectbox("ICP symptoms", [0,1], format_func=lambda x: yn[x])
focal = st.sidebar.selectbox("Focal symptoms", [0,1], format_func=lambda x: yn[x])
calc = st.sidebar.selectbox(">50% calcified", [0,1], index=1, format_func=lambda x: yn[x])
edema = st.sidebar.selectbox("Edema", [0,1], format_func=lambda x: yn[x])

row = make_row(age, size, loc, ep, ich, focal, calc, edema)


# -----------------------------------------------------------
# Predictions
# -----------------------------------------------------------
pA  = float(model_A.predict_proba(row)[0,1])
pB  = float(model_B.predict_proba(row)[0,1])
pAB = float(model_AB.predict_proba(row)[0,1])

binA  = lookup_bin(pA,  bins_A)
binB  = lookup_bin(pB,  bins_B)
binAB = lookup_bin(pAB, bins_AB)


# -----------------------------------------------------------
# Bin-level CI formatted output
# -----------------------------------------------------------
def format_bin_output(model_name, p, bin_data):
    if not bin_data:
        return f"**Risk:** {p*100:.1f}%\n\n_No calibration bin matched._"

    obs = bin_data["obs_rate"] * 100
    ci_low = bin_data["ci_low"] * 100
    ci_high = bin_data["ci_high"] * 100
    n = bin_data["n"]

    return f"""
**Risk:** {p*100:.1f}%

**Observed event rate:** {obs:.1f}%  
**95% calibration CI:** {ci_low:.1f}% – {ci_high:.1f}%  
**Bin size:** n={n}
"""


# -----------------------------------------------------------
# Display patient predictions
# -----------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Model A (trained on A)")
    st.markdown(format_bin_output("A", pA, binA))

with col2:
    st.subheader("Model B (trained on B)")
    st.markdown(format_bin_output("B", pB, binB))

with col3:
    st.subheader("Model AB (pooled A+B)")
    st.markdown(format_bin_output("AB", pAB, binAB))


# -----------------------------------------------------------
# Model performance summary
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Model Performance Summary")

st.markdown(
    f"""
### **Model A → B (trained on 507 A, validated on 110 B)**  
- **AUC:** {val_AtoB['auc']:.3f}  
- **Brier:** {val_AtoB['brier']:.3f}

### **Model B → A (trained on 110 B, validated on 507 A)**  
- **AUC:** {val_BtoA['auc']:.3f}  
- **Brier:** {val_BtoA['brier']:.3f}

### **Model AB (pooled A+B, 5-fold CV, N=617)**  
- **AUC:** {val_AB['auc']:.3f}  
- **Brier:** {val_AB['brier']:.3f}
"""
)


# -----------------------------------------------------------
# Calibration Table Display
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Calibration Tables")

def extract(bins):
    return pd.DataFrame(bins)[["mean_pred","obs_rate"]].rename(
        columns={"mean_pred":"Predicted","obs_rate":"Observed"}
    )

tabA, tabB, tabAB = st.tabs(["Model A", "Model B", "Model AB"])

with tabA: st.dataframe(extract(bins_A))
with tabB: st.dataframe(extract(bins_B))
with tabAB: st.dataframe(extract(bins_AB))

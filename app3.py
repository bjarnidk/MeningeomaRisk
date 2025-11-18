import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Load artifact
# -----------------------------
ARTIFACT_PATH = "meningioma_rf_models.joblib"
artifact = joblib.load(ARTIFACT_PATH)

model_A = artifact["model_A"]
model_AB = artifact["model_AB"]
feature_names = artifact["feature_names"]

# -----------------------------
# Location categories
# -----------------------------
location_map = {
    0: "infratentorial",
    1: "supratentorial",
    2: "skullbase",
    3: "convexity"
}

# -----------------------------
# Build feature row
# -----------------------------
def make_row(age, size_mm, loc_code, epilepsy, ich, focal, calcified, edema):
    row = {
        "age": age,
        "tumorsize": size_mm,
        "epilepsi": int(epilepsy),
        "tryksympt": int(ich),
        "focalsympt": int(focal),
        "calcified": int(calcified),
        "edema": int(edema),
    }

    df = pd.DataFrame([row])

    # Zero all location dummies
    for col in feature_names:
        if col.startswith("location_"):
            df[col] = 0

    chosen = f"location_{location_map[loc_code]}"
    if chosen in feature_names:
        df[chosen] = 1

    # Ensure all features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

# -----------------------------
# Find calibration bin
# -----------------------------
def lookup_bin(prob, bins):
    for b in bins:
        if b["p_min"] <= prob <= b["p_max"]:
            return b
    return None

# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide", page_title="Meningioma Intervention Risk")
st.title("Meningioma 15-Year Intervention Risk")

age = st.sidebar.number_input("Age", 18, 100, 65)
size = st.sidebar.number_input("Tumor size (mm)", 1, 150, 25)
loc = st.sidebar.selectbox("Location", list(location_map.keys()),
                           format_func=lambda x: location_map[x].capitalize())

yn = {0: "No", 1: "Yes"}
epilepsy = st.sidebar.selectbox("Epilepsy", [0,1], format_func=lambda x: yn[x])
ich = st.sidebar.selectbox("ICH symptoms", [0,1], format_func=lambda x: yn[x])
focal = st.sidebar.selectbox("Focal symptoms", [0,1], format_func=lambda x: yn[x])
calc = st.sidebar.selectbox(">50% calcified", [0,1], index=1, format_func=lambda x: yn[x])
edema = st.sidebar.selectbox("Edema", [0,1], format_func=lambda x: yn[x])

# -----------------------------
# Predictions
# -----------------------------
row = make_row(age, size, loc, epilepsy, ich, focal, calc, edema)

pA = float(model_A.predict_proba(row)[0,1])
pAB = float(model_AB.predict_proba(row)[0,1])

binA = lookup_bin(pA, artifact["validation_A"]["reliability_bins"])
binB = lookup_bin(pA, artifact["validation_B"]["reliability_bins"])
binAB = lookup_bin(pAB, artifact["validation_AB"]["reliability_bins"])

# -----------------------------
# Display
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Center A (trained on A only)")
    if binA:
        st.write(f"**Risk:** {pA*100:.1f}%")
        st.caption(f"Observed A: {binA['obs_rate']*100:.1f}% (n={binA['n']})")
    if binB:
        st.caption(f"Observed B (A→B): {binB['obs_rate']*100:.1f}% (n={binB['n']})")

with col2:
    st.subheader("Pooled A+B model")
    if binAB:
        st.write(f"**Risk:** {pAB*100:.1f}%")
        st.caption(f"Observed pooled: {binAB['obs_rate']*100:.1f}% (n={binAB['n']})")

valB = artifact["validation_B"]

st.markdown(
    f"""
### Model description and interpretation

This tool is based on a Random Forest classifier with isotonic calibration.  
The primary model was trained on **507 patients** from Center A and externally validated on  
**110 patients** from Center B, where it achieved:

- **AUC:** {valB['auc']:.3f}  
- **Brier score:** {valB['brier']:.3f}

Calibration was assessed by grouping patients into ten probability bins.  
Within each bin, predicted risks were compared to observed intervention rates,  
and 95% confidence intervals were calculated using Wilson’s method.

The width of these intervals reflects how many patients contributed to the bin.  

For each patient entered, the app displays:

- Predicted probability of intervention within 15 years (Center A and pooled models)  
- Observed event rates from calibration/validation cohorts  

 The table below shows the probability bins, where model prediction and observed events can be evaluated side-by-side.
"""
)


# -----------------------------
# Calibration tables
# -----------------------------
st.markdown("---")
st.subheader("Calibration tables")

tabA, tabB, tabAB = st.tabs(["Center A (internal)", "Center B (A→B)", "Pooled A+B"])

def extract(df): return pd.DataFrame(df)[["mean_pred", "obs_rate"]].rename(
    columns={"mean_pred": "Predicted", "obs_rate": "Observed"})

with tabA: st.dataframe(extract(artifact["validation_A"]["reliability_bins"]))
with tabB: st.dataframe(extract(artifact["validation_B"]["reliability_bins"]))
with tabAB: st.dataframe(extract(artifact["validation_AB"]["reliability_bins"]))

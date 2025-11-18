import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Load artifact
# -----------------------------
artifact = joblib.load("meningioma_models.joblib")

model_A  = artifact["model_A"]
model_B  = artifact["model_B"]
model_AB = artifact["model_AB"]
feature_names = artifact["feature_names"]

bins_A  = artifact["reliability_A"]
bins_B  = artifact["reliability_B"]
bins_AB = artifact["reliability_AB"]


# -----------------------------
# Location mapping
# -----------------------------
location_map = {
    0: "infratentorial",
    1: "supratentorial",
    2: "skullbase",
    3: "convexity",
}


# -----------------------------
# Build row
# -----------------------------
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

    # zero location
    for c in feature_names:
        if c.startswith("location_"):
            df[c] = 0

    col = f"location_{location_map[loc_code]}"
    if col in feature_names:
        df[col] = 1

    # ensure all columns
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0

    return df[feature_names]


# -----------------------------
# Locate calibration bin
# -----------------------------
def lookup_bin(p, bins):
    for b in bins:
        if b["p_min"] <= p <= b["p_max"]:
            return b
    return None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Meningioma Risk", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")

age = st.sidebar.number_input("Age", 18, 100, 65)
size = st.sidebar.number_input("Tumor size (mm)", 1, 150, 25)
loc = st.sidebar.selectbox("Location", list(location_map.keys()),
                           format_func=lambda x: location_map[x].capitalize())

yn = {0: "No", 1: "Yes"}
ep = st.sidebar.selectbox("Epilepsy", [0,1], format_func=lambda x: yn[x])
ich = st.sidebar.selectbox("ICH symptoms", [0,1], format_func=lambda x: yn[x])
focal = st.sidebar.selectbox("Focal symptoms", [0,1], format_func=lambda x: yn[x])
calc = st.sidebar.selectbox(">50% calcified", [0,1], index=1, format_func=lambda x: yn[x])
edema = st.sidebar.selectbox("Edema", [0,1], format_func=lambda x: yn[x])

row = make_row(age, size, loc, ep, ich, focal, calc, edema)


# -----------------------------
# Predictions
# -----------------------------
pA  = float(model_A.predict_proba(row)[0,1])
pB  = float(model_B.predict_proba(row)[0,1])
pAB = float(model_AB.predict_proba(row)[0,1])

binA  = lookup_bin(pA,  bins_A)
binB  = lookup_bin(pB,  bins_B)
binAB = lookup_bin(pAB, bins_AB)


# -----------------------------
# Display
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Model A (trained on A)")
    st.write(f"**Risk:** {pA*100:.1f}%")
    if binA:
        st.caption(f"Observed A: {binA['obs_rate']*100:.1f}% (n={binA['n']})")

with col2:
    st.subheader("Model B (trained on B)")
    st.write(f"**Risk:** {pB*100:.1f}%")
    if binB:
        st.caption(f"Observed B: {binB['obs_rate']*100:.1f}% (n={binB['n']})")

with col3:
    st.subheader("Model AB (trained on A+B)")
    st.write(f"**Risk:** {pAB*100:.1f}%")
    if binAB:
        st.caption(f"Observed pooled: {binAB['obs_rate']*100:.1f}% (n={binAB['n']})")


# -----------------------------
# Calibration tables
# -----------------------------
st.markdown("---")
st.subheader("Calibration tables")

def extract(bins):
    return pd.DataFrame(bins)[["mean_pred","obs_rate"]].rename(
        columns={"mean_pred":"Predicted","obs_rate":"Observed"}
    )

tabA, tabB, tabAB = st.tabs(["Model A", "Model B", "Model AB"])

with tabA:  st.dataframe(extract(bins_A))
with tabB:  st.dataframe(extract(bins_B))
with tabAB: st.dataframe(extract(bins_AB))

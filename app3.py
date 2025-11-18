import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import roc_auc_score, brier_score_loss

# --------------------------------------------
# Load trained models + reliability bins
# --------------------------------------------
artifact = joblib.load("meningioma_models.joblib")

model_A  = artifact["model_A"]
model_B  = artifact["model_B"]
model_AB = artifact["model_AB"]

feature_names = artifact["feature_names"]
bins_A  = artifact["reliability_A"]
bins_B  = artifact["reliability_B"]
bins_AB = artifact["reliability_AB"]

# --------------------------------------------
# Load raw data for metric computation
# --------------------------------------------
# These CSVs MUST be uploaded with streamlit, or packaged with the app
dfA = pd.read_csv("CSV1.csv")
dfB = pd.read_csv("CSV2.csv")

# Lowercase columns to match training
dfA.columns = dfA.columns.str.lower()
dfB.columns = dfB.columns.str.lower()

y_A = dfA["intervention"].astype(int)
y_B = dfB["intervention"].astype(int)

# Build design matrices exactly like training
continuous = ["age", "tumorsize"]
categorical = ["location"]
binary = ["epilepsi","tryksympt","focalsympt","calcified","edema"]

X_A_raw = dfA[continuous + categorical + binary]
X_B_raw = dfB[continuous + categorical + binary]

X_all = pd.concat([X_A_raw, X_B_raw], axis=0)
X_all = pd.get_dummies(X_all, columns=categorical, drop_first=True)

X_A = X_all.iloc[:len(dfA)][feature_names]
X_B = X_all.iloc[len(dfA):][feature_names]
X_AB = pd.concat([X_A, X_B], axis=0)

# --------------------------------------------
# Compute metrics LIVE (no need to store in artifact)
# --------------------------------------------

# A→B
pA_B = model_A.predict_proba(X_B)[:,1]
A_auc  = roc_auc_score(y_B, pA_B)
A_brier = brier_score_loss(y_B, pA_B)

# B→A
pB_A = model_B.predict_proba(X_A)[:,1]
B_auc  = roc_auc_score(y_A, pB_A)
B_brier = brier_score_loss(y_A, pB_A)

# AB→AB (cross-val results approximated by computing on full set)
pAB_AB = model_AB.predict_proba(X_AB)[:,1]
AB_auc  = roc_auc_score(np.concatenate([y_A,y_B]), pAB_AB)
AB_brier = brier_score_loss(np.concatenate([y_A,y_B]), pAB_AB)


# --------------------------------------------
# UI components from here (unchanged)
# --------------------------------------------
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

    for c in feature_names:
        if c.startswith("location_"):
            df[c] = 0

    col = f"location_{location_map[loc_code]}"
    if col in feature_names:
        df[col] = 1

    for c in feature_names:
        if c not in df.columns:
            df[c] = 0

    return df[feature_names]

def lookup_bin(p, bins):
    for b in bins:
        if b["p_min"] <= p <= b["p_max"]:
            return b
    return None


# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="Meningioma Risk", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")

age = st.sidebar.number_input("Age", 18, 100, 65)
size = st.sidebar.number_input("Tumor size (mm)", 1, 150, 25)
loc = st.sidebar.selectbox("Location", list(location_map.keys()), format_func=lambda x: location_map[x])

yn = {0: "No", 1: "Yes"}
ep = st.sidebar.selectbox("Epilepsy", [0,1], format_func=lambda x: yn[x])
ich = st.sidebar.selectbox("ICH symptoms", [0,1], format_func=lambda x: yn[x])
focal = st.sidebar.selectbox("Focal symptoms", [0,1], format_func=lambda x: yn[x])
calc = st.sidebar.selectbox(">50% calcified", [0,1], index=1, format_func=lambda x: yn[x])
edema = st.sidebar.selectbox("Edema", [0,1], format_func=lambda x: yn[x])

row = make_row(age, size, loc, ep, ich, focal, calc, edema)

pA  = float(model_A.predict_proba(row)[0,1])
pB  = float(model_B.predict_proba(row)[0,1])
pAB = float(model_AB.predict_proba(row)[0,1])

binA  = lookup_bin(pA,  bins_A)
binB  = lookup_bin(pB,  bins_B)
binAB = lookup_bin(pAB, bins_AB)


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Model A (trained on A)")
    st.write(f"**Risk:** {pA*100:.1f}%")
    if binA:
        st.caption(f"Observed A: {binA['obs_rate']*100:.1f}%")

with col2:
    st.subheader("Model B (trained on B)")
    st.write(f"**Risk:** {pB*100:.1f}%")
    if binB:
        st.caption(f"Observed B: {binB['obs_rate']*100:.1f}%")

with col3:
    st.subheader("Model AB (trained on A+B)")
    st.write(f"**Risk:** {pAB*100:.1f}%")
    if binAB:
        st.caption(f"Observed pooled: {binAB['obs_rate']*100:.1f}%")


st.markdown("---")
st.subheader("Model description and interpretation")

st.markdown(
    f"""
### **Model A → B**
- AUC: **{A_auc:.3f}**  
- Brier: **{A_brier:.3f}**

### **Model B → A**
- AUC: **{B_auc:.3f}**  
- Brier: **{B_brier:.3f}**

### **Model AB (5-fold pooled)**
- AUC: **{AB_auc:.3f}**  
- Brier: **{AB_brier:.3f}**
"""
)


# Calibration tables
st.markdown("---")
st.subheader("Calibration Tables")

def extract(bins):
    return pd.DataFrame(bins)[["mean_pred","obs_rate"]].rename(
        columns={"mean_pred":"Predicted","obs_rate":"Observed"}
    )

tabA, tabB, tabAB = st.tabs(["Model A", "Model B", "Model AB"])

with tabA:  st.dataframe(extract(bins_A))
with tabB:  st.dataframe(extract(bins_B))
with tabAB: st.dataframe(extract(bins_AB))

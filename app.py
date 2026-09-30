import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="DIM Insight 360",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {min-width: 285px; max-width: 285px;}
.small-note {font-size: 0.82rem; opacity: .72;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_data(n=30000, seed=42):
    rng = np.random.default_rng(seed)

    services = [
        "Médecine interne", "Cardiologie", "Chirurgie", "Urgences",
        "Pneumologie", "Gériatrie", "Neurologie", "Oncologie"
    ]
    diagnoses = {
        "I10": "Hypertension essentielle",
        "I50": "Insuffisance cardiaque",
        "J18": "Pneumopathie",
        "E11": "Diabète type 2",
        "C50": "Tumeur maligne du sein",
        "G40": "Épilepsie",
        "K35": "Appendicite aiguë",
        "N39": "Infection urinaire",
    }
    ccam_codes = ["DEQP003", "DZQM006", "HHFA001", "JAFA001", "QZFA001", "YYYY001"]

    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2026-09-30")
    seconds = rng.integers(start.value // 10**9, end.value // 10**9, n)
    dates = pd.to_datetime(seconds, unit="s")

    service = rng.choice(services, n, p=[.14,.13,.12,.18,.10,.12,.10,.11])
    sex = rng.choice(["F", "M"], n)
    age = np.clip(rng.normal(61, 20, n).round(), 0, 100).astype(int)
    admission = rng.choice(["Urgence", "Programmée", "Transfert"], n, p=[.56,.34,.10])
    mode = rng.choice(["HC", "HDJ"], n, p=[.77,.23])

    base_los = rng.gamma(2.2, 2.2, n)
    service_effect = pd.Series(service).map({
        "Urgences": -1.2, "Chirurgie": 0.8, "Gériatrie": 2.2,
        "Oncologie": 1.2, "Cardiologie": 0.6
    }).fillna(0).to_numpy()
    age_effect = np.maximum(age - 70, 0) * 0.04
    los = np.clip(np.round(base_los + service_effect + age_effect), 0, 30).astype(int)
    los = np.where(mode == "HDJ", rng.choice([0, 1], n, p=[.75, .25]), np.maximum(los, 1))

    discharge = dates + pd.to_timedelta(los, unit="D")
    diag_code = rng.choice(list(diagnoses.keys()), n)
    diag_label = pd.Series(diag_code).map(diagnoses).to_numpy()
    ccam = rng.choice(ccam_codes, n)
    comorb = np.clip(rng.poisson(1.6, n), 0, 8)

    wait = np.clip(
        rng.normal(95, 55, n)
        + np.where(admission == "Urgence", 35, -20)
        + np.where(service == "Urgences", 30, 0),
        5, 420
    ).round().astype(int)

    occupancy = np.clip(
        rng.normal(.82, .08, n)
        + np.where(service == "Urgences", .06, 0)
        + np.where(service == "Gériatrie", .04, 0),
        .45, 1.15
    )

    cost = np.clip(
        650 + los * rng.normal(520, 90, n) + comorb * 260 + rng.normal(0, 450, n),
        300, 30000
    ).round(2)
    revenue = (cost * rng.uniform(1.05, 1.35, n)).round(2)

    readmission = (
        rng.random(n)
        < np.clip(.06 + comorb * .025 + (los > 10) * .05 + (age > 75) * .03, .03, .35)
    ).astype(int)

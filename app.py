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

    df = pd.DataFrame({
        "patient_id": [f"P{100000+i}" for i in range(n)],
        "date_admission": dates,
        "date_sortie": discharge,
        "service": service,
        "sexe": sex,
        "age": age,
        "type_admission": admission,
        "mode_sejour": mode,
        "diagnostic_principal": diag_code,
        "libelle_diagnostic": diag_label,
        "acte_ccam": ccam,
        "nb_comorbidites": comorb,
        "dms": los,
        "temps_attente_min": wait,
        "taux_occupation": occupancy,
        "cout_simule": cost,
        "recette_simulee": revenue,
        "payeur": rng.choice(["Assurance Maladie", "Mutuelle", "Autre"], n, p=[.76,.20,.04]),
        "readmission_30j": readmission,
    })

    # Anomalies synthétiques pour le module qualité
    idx = df.index.to_numpy()
    for frac, col in [(0.015, "diagnostic_principal"), (0.012, "acte_ccam")]:
        sel = rng.choice(idx, size=max(1, int(n*frac)), replace=False)
        df.loc[sel, col] = np.nan

    sel = rng.choice(idx, size=max(1, int(n*.003)), replace=False)
    df.loc[sel, "age"] = 125

    sel = rng.choice(idx, size=max(1, int(n*.002)), replace=False)
    df.loc[sel, "date_sortie"] = df.loc[sel, "date_admission"] - pd.Timedelta(days=1)

    df["mois"] = df["date_admission"].dt.to_period("M").astype(str)
    return df


@st.cache_resource(show_spinner=False)
def train_model(df_model):
    d = df_model.copy()
    d["long_sejour"] = (d["dms"] >= 8).astype(int)

    features = [
        "age", "nb_comorbidites", "temps_attente_min",
        "taux_occupation", "service", "type_admission", "mode_sejour"
    ]
    X = d[features].copy()
    encoders = {}

    for col in ["service", "type_admission", "mode_sejour"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    y = d["long_sejour"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
    }
    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    return model, encoders, metrics, importance


def header(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.info("Données entièrement synthétiques — projet démonstrateur, sans données patient réelles.")


def show_kpis(data):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Séjours", f"{len(data):,}".replace(",", " "))
    c2.metric("DMS", f"{data['dms'].mean():.1f} j" if len(data) else "—")
    c3.metric("Occupation", f"{data['taux_occupation'].mean()*100:.1f}%" if len(data) else "—")
    c4.metric("Attente moyenne", f"{data['temps_attente_min'].mean():.0f} min" if len(data) else "—")
    c5.metric("Réadmission 30j", f"{data['readmission_30j'].mean()*100:.1f}%" if len(data) else "—")


df = generate_data()

st.sidebar.title("🏥 DIM Insight 360")
st.sidebar.caption("Hospital Data Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue Direction",
        "Activité hospitalière",
        "PMSI & codage",
        "Qualité des données",
        "Flux patients",
        "Médico-économique",
        "Prédiction séjours longs",
        "Prévisions",
        "Assistant analytique",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")

min_date = df["date_admission"].min().date()
max_date = df["date_admission"].max().date()

date_range = st.sidebar.date_input(
    "Période",
    value=(max_date - timedelta(days=180), max_date),
    min_value=min_date,
    max_value=max_date,
)

services = st.sidebar.multiselect(
    "Service", sorted(df["service"].unique()), default=[]
)
sexes = st.sidebar.multiselect("Sexe", ["F", "M"], default=[])

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    d1 = pd.Timestamp(date_range[0])
    d2 = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    d1 = pd.Timestamp(min_date)
    d2 = pd.Timestamp(max_date) + pd.Timedelta(days=1)

f = df[(df["date_admission"] >= d1) & (df["date_admission"] < d2)].copy()
if services:
    f = f[f["service"].isin(services)]
if sexes:
    f = f[f["sexe"].isin(sexes)]

st.sidebar.caption(f"{len(f):,} séjours sélectionnés".replace(",", " "))

if f.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

if page == "Vue Direction":
    header("📊 Vue Direction", "Synthèse des principaux indicateurs hospitaliers")
    show_kpis(f)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Activité mensuelle")
        st.line_chart(f.groupby("mois").size().rename("Séjours"))
    with c2:
        st.subheader("Séjours par service")
        st.bar_chart(f["service"].value_counts().sort_values())

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Répartition HC / HDJ")
        st.bar_chart(f["mode_sejour"].value_counts())
    with c4:
        st.subheader("Top diagnostics")
        st.bar_chart(f["libelle_diagnostic"].value_counts().head(8))

    st.subheader("Alertes de pilotage")
    alerts = []
    occ = f.groupby("service")["taux_occupation"].mean()
    wait = f.groupby("service")["temps_attente_min"].mean()
    for srv, value in occ.items():
        if value > .95:
            alerts.append(f"⚠️ {srv} : taux d'occupation élevé ({value*100:.1f}%).")
    for srv, value in wait.items():
        if value > 140:
            alerts.append(f"⏱️ {srv} : attente moyenne élevée ({value:.0f} min).")
    if alerts:
        for a in alerts[:8]:
            st.warning(a)
    else:
        st.success("Aucune alerte majeure sur la période sélectionnée.")

elif page == "Activité hospitalière":
    header("🏨 Activité hospitalière", "Admissions, séjours et durées de prise en charge")
    show_kpis(f)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Admissions par mois")
        st.line_chart(f.groupby("mois").size())
    with c2:
        st.subheader("Types d'admission")
        st.bar_chart(f["type_admission"].value_counts())

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("DMS moyenne par service")
        st.bar_chart(f.groupby("service")["dms"].mean().sort_values())
    with c4:
        st.subheader("Âge moyen par service")
        st.bar_chart(f.groupby("service")["age"].mean().sort_values())

    st.subheader("Détail des séjours")
    cols = ["patient_id","date_admission","service","age","sexe","type_admission","mode_sejour","dms"]
    st.dataframe(
        f[cols].sort_values("date_admission", ascending=False).head(500),
        use_container_width=True
    )

elif page == "PMSI & codage":
    header("🧾 PMSI & codage", "Exploration du codage diagnostique et des actes — simulation pédagogique")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diagnostics renseignés", f"{f['diagnostic_principal'].notna().mean()*100:.1f}%")
    c2.metric("Actes CCAM renseignés", f"{f['acte_ccam'].notna().mean()*100:.1f}%")
    c3.metric("Comorbidités moyennes", f"{f['nb_comorbidites'].mean():.1f}")
    c4.metric("HDJ", f"{(f['mode_sejour']=='HDJ').mean()*100:.1f}%")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Diagnostics principaux")
        diag = (
            f.groupby(["diagnostic_principal","libelle_diagnostic"])
            .size().sort_values(ascending=False).head(12)
            .rename("Séjours").reset_index()
        )
        st.dataframe(diag, use_container_width=True)
    with c6:
        st.subheader("Actes CCAM les plus fréquents")
        st.bar_chart(f["acte_ccam"].value_counts().head(10))

    checks = pd.DataFrame({
        "Contrôle": [
            "Diagnostic principal manquant",
            "Acte CCAM manquant",
            "Date sortie < admission",
            "Âge hors plage plausible",
            "Séjour HC de durée nulle",
        ],
        "Nombre": [
            f["diagnostic_principal"].isna().sum(),
            f["acte_ccam"].isna().sum(),
            (f["date_sortie"] < f["date_admission"]).sum(),
            ((f["age"] < 0) | (f["age"] > 110)).sum(),
            ((f["mode_sejour"] == "HC") & (f["dms"] <= 0)).sum(),
        ],
    })
    checks["Taux (%)"] = checks["Nombre"] / max(len(f), 1) * 100
    st.subheader("Contrôles de cohérence PMSI simulés")
    st.dataframe(checks, use_container_width=True)

elif page == "Qualité des données":
    header("✅ Qualité des données", "Contrôle d'exhaustivité, cohérence et anomalies")

    missing_diag = f["diagnostic_principal"].isna().sum()
    missing_ccam = f["acte_ccam"].isna().sum()
    bad_dates = (f["date_sortie"] < f["date_admission"]).sum()
    bad_age = ((f["age"] < 0) | (f["age"] > 110)).sum()
    duplicates = f.duplicated(subset=["patient_id","date_admission"]).sum()

    total_checks = max(len(f) * 5, 1)
    issues = missing_diag + missing_ccam + bad_dates + bad_age + duplicates
    quality_score = max(0, 100 - issues / total_checks * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score qualité", f"{quality_score:.1f}%")
    c2.metric("Diagnostics manquants", str(missing_diag))
    c3.metric("Actes manquants", str(missing_ccam))
    c4.metric("Anomalies dates/âge", str(bad_dates + bad_age))

    quality_table = pd.DataFrame({
        "Règle": [
            "Diagnostic principal non vide",
            "Acte CCAM non vide",
            "Date sortie >= admission",
            "Âge entre 0 et 110 ans",
            "Pas de doublon séjour",
        ],
        "Anomalies": [missing_diag, missing_ccam, bad_dates, bad_age, duplicates],
    })
    quality_table["Conformité (%)"] = 100 - quality_table["Anomalies"] / max(len(f), 1) * 100
    st.dataframe(quality_table, use_container_width=True)

    anomalies = f[
        f["diagnostic_principal"].isna()
        | f["acte_ccam"].isna()
        | (f["date_sortie"] < f["date_admission"])
        | (f["age"] > 110)
    ]
    st.subheader("Échantillon d'anomalies")
    st.dataframe(anomalies.head(300), use_container_width=True)
    st.download_button(
        "⬇️ Exporter les anomalies en CSV",
        anomalies.to_csv(index=False).encode("utf-8"),
        file_name="anomalies_dim_insight.csv",
        mime="text/csv",
    )

elif page == "Flux patients":
    header("🚑 Flux patients", "Temps d'attente, congestion et occupation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attente moyenne", f"{f['temps_attente_min'].mean():.0f} min")
    c2.metric("Attente > 2h", f"{(f['temps_attente_min']>120).mean()*100:.1f}%")
    c3.metric("Occupation moyenne", f"{f['taux_occupation'].mean()*100:.1f}%")
    c4.metric("Occupation > 95%", f"{(f['taux_occupation']>.95).mean()*100:.1f}%")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Temps d'attente par service")
        st.bar_chart(f.groupby("service")["temps_attente_min"].mean().sort_values())
    with c6:
        st.subheader("Taux d'occupation par service")
        st.bar_chart((f.groupby("service")["taux_occupation"].mean()*100).sort_values())

    congestion = f.groupby("service").agg(
        attente=("temps_attente_min","mean"),
        occupation=("taux_occupation","mean"),
        sejours=("patient_id","count"),
    )
    congestion["score_congestion"] = (
        congestion["attente"]/max(congestion["attente"].max(),1)*50
        + congestion["occupation"]/max(congestion["occupation"].max(),1)*50
    )
    st.subheader("Indice de congestion")
    st.dataframe(
        congestion.sort_values("score_congestion", ascending=False).round(2),
        use_container_width=True,
    )

elif page == "Médico-économique":
    header("💶 Analyse médico-économique", "Coûts, recettes simulées et performance par service")

    total_cost = f["cout_simule"].sum()
    total_rev = f["recette_simulee"].sum()
    margin = total_rev - total_cost

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coût simulé total", f"{total_cost/1e6:.2f} M€")
    c2.metric("Recettes simulées", f"{total_rev/1e6:.2f} M€")
    c3.metric("Marge simulée", f"{margin/1e6:.2f} M€")
    c4.metric("Coût moyen / séjour", f"{f['cout_simule'].mean():,.0f} €")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Coût moyen par service")
        st.bar_chart(f.groupby("service")["cout_simule"].mean().sort_values())
    with c6:
        st.subheader("Recette moyenne par service")
        st.bar_chart(f.groupby("service")["recette_simulee"].mean().sort_values())

    eco = f.groupby("service").agg(
        sejours=("patient_id","count"),
        dms=("dms","mean"),
        cout_moyen=("cout_simule","mean"),
        recette_moyenne=("recette_simulee","mean"),
        recette_totale=("recette_simulee","sum"),
        cout_total=("cout_simule","sum"),
    )
    eco["marge_totale"] = eco["recette_totale"] - eco["cout_total"]
    st.subheader("Synthèse par service")
    st.dataframe(eco.round(2), use_container_width=True)

elif page == "Prédiction séjours longs":
    header("🤖 Prédiction des séjours longs", "Random Forest sur données synthétiques")

    model, encoders, metrics, importance = train_model(df)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("ROC-AUC", f"{metrics['auc']:.3f}")

    st.subheader("Importance des variables")
    st.bar_chart(importance)

    st.subheader("Simulation individuelle")
    c3, c4, c5 = st.columns(3)
    age_v = c3.slider("Âge", 0, 100, 67)
    comorb_v = c4.slider("Nombre de comorbidités", 0, 8, 2)
    wait_v = c5.slider("Temps d'attente (min)", 5, 420, 90)

    c6, c7, c8 = st.columns(3)
    service_v = c6.selectbox("Service", encoders["service"].classes_)
    admission_v = c7.selectbox("Type d'admission", encoders["type_admission"].classes_)
    mode_v = c8.selectbox("Mode de séjour", encoders["mode_sejour"].classes_)
    occ_v = st.slider("Taux d'occupation", 0.45, 1.15, 0.85, 0.01)

    X_new = pd.DataFrame([{
        "age": age_v,
        "nb_comorbidites": comorb_v,
        "temps_attente_min": wait_v,
        "taux_occupation": occ_v,
        "service": encoders["service"].transform([service_v])[0],
        "type_admission": encoders["type_admission"].transform([admission_v])[0],
        "mode_sejour": encoders["mode_sejour"].transform([mode_v])[0],
    }])

    risk = model.predict_proba(X_new)[0, 1]
    st.metric("Probabilité simulée de séjour ≥ 8 jours", f"{risk*100:.1f}%")
    if risk >= .65:
        st.warning("Risque simulé élevé — démonstration uniquement.")
    elif risk >= .35:
        st.info("Risque simulé intermédiaire.")
    else:
        st.success("Risque simulé faible.")

elif page == "Prévisions":
    header("📈 Prévisions d'activité", "Projection simple des admissions")

    daily = (
        df.groupby(df["date_admission"].dt.date)
        .size().rename("admissions").reset_index()
    )
    daily["date_admission"] = pd.to_datetime(daily["date_admission"])
    daily = daily.sort_values("date_admission")
    daily["t"] = np.arange(len(daily))

    lr = LinearRegression().fit(daily[["t"]], daily["admissions"])
    horizon = st.slider("Horizon de prévision (jours)", 7, 60, 30)

    future_t = np.arange(len(daily), len(daily)+horizon)
    future_dates = pd.date_range(
        daily["date_admission"].max() + pd.Timedelta(days=1),
        periods=horizon,
    )
    pred = np.maximum(0, lr.predict(future_t.reshape(-1,1)))

    forecast = pd.DataFrame({
        "date": future_dates,
        "admissions_prévues": pred,
    })

    st.subheader("Historique récent")
    st.line_chart(daily.tail(90).set_index("date_admission")["admissions"])
    st.subheader("Prévision")
    st.line_chart(forecast.set_index("date")["admissions_prévues"])
    st.caption("Modèle volontairement simple pour démonstration. Pas destiné à un usage clinique ou opérationnel réel.")

elif page == "Assistant analytique":
    header("💬 Assistant analytique", "Questions simples sur les données filtrées")

    st.markdown("""
    Exemples :
    - **Quel service a la DMS la plus élevée ?**
    - **Quel service a le plus de séjours ?**
    - **Quel est le taux de réadmission ?**
    - **Quel service a l'attente la plus longue ?**
    - **Combien de diagnostics sont manquants ?**
    - **Quel service a le taux d'occupation le plus élevé ?**
    """)

    q = st.text_input("Posez une question")

    if q:
        ql = q.lower()

        if "dms" in ql or "durée" in ql:
            s = f.groupby("service")["dms"].mean().sort_values(ascending=False)
            st.success(f"Le service avec la DMS moyenne la plus élevée est **{s.index[0]}** avec **{s.iloc[0]:.1f} jours**.")

        elif "plus de séjour" in ql or "plus de sejour" in ql or "volume" in ql:
            s = f["service"].value_counts()
            st.success(f"Le plus grand volume est observé en **{s.index[0]}** avec **{int(s.iloc[0])} séjours**.")

        elif "réadmission" in ql or "readmission" in ql:
            st.success(f"Le taux de réadmission à 30 jours est de **{f['readmission_30j'].mean()*100:.1f}%**.")

        elif "attente" in ql:
            s = f.groupby("service")["temps_attente_min"].mean().sort_values(ascending=False)
            st.success(f"L'attente moyenne la plus élevée est observée en **{s.index[0]}** : **{s.iloc[0]:.0f} minutes**.")

        elif "diagnostic" in ql and ("manquant" in ql or "manquants" in ql):
            n = int(f["diagnostic_principal"].isna().sum())
            st.success(f"Il y a **{n} diagnostics principaux manquants** sur la période filtrée.")

        elif "occupation" in ql:
            s = f.groupby("service")["taux_occupation"].mean().sort_values(ascending=False)
            st.success(f"Le taux d'occupation moyen le plus élevé est observé en **{s.index[0]}** : **{s.iloc[0]*100:.1f}%**.")

        else:
            st.info("Je peux répondre aux questions sur la DMS, les volumes, les réadmissions, les attentes, les diagnostics manquants et l'occupation.")

st.markdown("---")
st.caption("DIM Insight 360 — Projet démonstrateur sur données synthétiques. Aucun usage clinique réel.")

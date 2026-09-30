import streamlit as st

st.set_page_config(
    page_title="DIM Insight 360",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 DIM Insight 360")
st.subheader("Hospital Data Intelligence Platform")

st.success("✅ Application déployée avec succès")

st.write(
    """
    Plateforme démonstratrice d'analyse hospitalière,
    de pilotage DIM, de qualité des données et
    d'intelligence artificielle.
    """
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Séjours analysés", "25 430")
col2.metric("DMS", "5.8 jours")
col3.metric("Taux d'occupation", "83 %")
col4.metric("Qualité des données", "96.4 %")

st.info("Données synthétiques — projet démonstrateur.")

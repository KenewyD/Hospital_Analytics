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

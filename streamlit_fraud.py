import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "rf_fraud_model.joblib"

st.set_page_config(page_title="Kreditkartenbetrug Klassifikator", layout="wide")
st.title("Kreditkartenbetrug – Klassifikator")

# Modell laden
if not os.path.exists(MODEL_PATH):
    st.error(f"Modell '{MODEL_PATH}' nicht gefunden. Lege die Datei in diesen Ordner.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved.get("model")
scaler = saved.get("scaler")
features = saved.get("features")

if model is None or scaler is None or features is None:
    st.error("Modelldatei unvollständig: erwartet Keys 'model', 'scaler', 'features'.")
    st.stop()

st.markdown("**Eingabeoptionen:** CSV-Datei hochladen *oder* Textzeilen mit Komma-Werten in Feature-Reihenfolge.")

mode = st.radio("Eingabe wählen", ["CSV-Datei", "Text"], horizontal=True)
df = None

if mode == "CSV-Datei":
    up = st.file_uploader("CSV hochladen", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"CSV konnte nicht gelesen werden: {e}")
else:
    txt = st.text_area(
        "Jede Zeile = eine Transaktion, Werte mit Komma trennen",
        height=220,
        placeholder="z.B.\n-1.3598,1.1919,-0.358..., ...\n0.5299,-0.1483,0.207..., ...",
    )
    if txt.strip():
        try:
            rows = [[float(p.strip()) for p in line.split(",")]
                    for line in txt.strip().splitlines() if line.strip()]
            df = pd.DataFrame(rows, columns=features)
        except ValueError:
            st.error("Ungültige Werte: Bitte nur Zahlen verwenden.")
        except Exception as e:
            st.error(f"Eingabeproblem: {e}")

if df is not None:
    # Falls CSV: sicherstellen, dass die Spalten passen
    if mode == "CSV-Datei":
        missing = [c for c in features if c not in df.columns]
        extra = [c for c in df.columns if c not in features]
        if missing:
            st.error(f"Fehlende Spalten in CSV: {missing}")
            st.stop()
        if extra:
            st.info(f"Ignoriere zusätzliche Spalten: {extra}")
        X = df[features]
    else:
        X = df[features]

    try:
        Xs = scaler.transform(X)
        preds = model.predict(Xs)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xs)[:, 1]
        else:
            # Fallback, falls kein predict_proba vorhanden
            # Normiere Decision Function grob auf 0..1
            if hasattr(model, "decision_function"):
                z = model.decision_function(Xs)
                probs = 1 / (1 + np.exp(-z))
            else:
                probs = np.zeros(len(preds))
        out = X.copy()
        out["Prediction"] = preds
        out["Fraud_Probability"] = probs
        st.success(f"{len(out)} Zeile(n) klassifiziert.")
        st.dataframe(out, use_container_width=True)
        st.download_button("Ergebnisse als CSV herunterladen",
                           out.to_csv(index=False), "fraud_predictions.csv")
    except Exception as e:
        st.error(f"Vorhersage fehlgeschlagen: {e}")
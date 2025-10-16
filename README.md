# Credit Card Fraud – Projektüberblick

Eine kompakte Sammlung zur **Analyse**, **Modellierung** und **Prüfung** von Kreditkartenbetrug.

---

## 📁 Projektstruktur

```
plots/                                  # Sämtliche Diagramme (EDA/Modelldiagnostik)
creditcard.csv                          # Datensatz (Kaggle-Link unten)
credit_card_fraud_eda_and_model.py      # Datenanalyse & TRAINING
fraud_gui.py                            # (Tkinter) frühere GUI-Variante – auf macOS problematisch
streamlit_fraud.py                      # Empfohlene Browser-UI für die Klassifikation
rf_fraud_model.joblib                   # Trainiertes Modell (model, scaler, features)
```

**Datensatz:**  
[Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

---

## 🚀 Starten der Apps

### 1) Empfohlen: Browser-UI (Streamlit)
Ohne native GUI‑Abhängigkeiten; läuft im Browser.

**Installation (einmalig):**
```bash
python3 -m pip install --upgrade streamlit pandas numpy scikit-learn joblib
```

**Starten:**  
- **macOS:**  
  ```bash
  python3 -m streamlit run streamlit_fraud.py
  ```
- **Windows / Linux (oft ebenfalls möglich):**  
  ```bash
  streamlit run streamlit_fraud.py
  ```

> **Hinweis:** `streamlit_fraud.py` erwartet, dass `rf_fraud_model.joblib` im selben Ordner liegt und die Keys
> `model`, `scaler`, `features` enthält.

**Eingabeoptionen in der App:**
- CSV‑Upload (Spaltennamen müssen `features` entsprechen), **oder**
- Textzeilen: je Zeile eine Transaktion, Werte komma‑separiert in **Feature‑Reihenfolge**.

---

## 📌 Beispieldatensatz (kopierbar)

Nutze die folgende **eine Zeile** direkt in der Text‑Eingabe der App.  
Sie enthält **V1..V28 + Amount** (also **ohne** `Time`, **ohne** `Class`).

```
-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62
```

> **Wichtig:**  
> - Wenn dein Roh‑CSV vorne eine **laufende Index‑Spalte** hat → **entfernen**.  
> - Steht hinten die **Klasse/`Class`** → **entfernen** (auch das Komma).  
> - Bei CSV‑Upload müssen die Spaltennamen exakt den `features` entsprechen.

---

### 2) (Alt) Tkinter‑GUI
`fraud_gui.py` war die ursprüngliche Desktop‑App. Auf macOS kann es zu Tk/Tcl‑Konflikten kommen.  
Nutze stattdessen die Streamlit‑Variante, wenn du keine Systemanpassungen vornehmen willst.

---

## 🧠 Training & EDA

Das Skript `credit_card_fraud_eda_and_model.py` führt **Explorative Datenanalyse** und das **Training** durch.  
Das erzeugte Modell wird als `rf_fraud_model.joblib` gespeichert und später von der UI verwendet.

> ⚠️ **Leistungshinweis:** Auf schwächeren Rechnern kann das Training länger dauern.  
> Beispiel: ~3 Minuten mit hoher CPU‑Auslastung.

**Ausgabe/Artefakte:**
- `plots/` enthält Diagramme (z. B. Verteilungen, Korrelationsmatrizen, ROC/PR‑Kurven, Feature‑Importance).
- `rf_fraud_model.joblib` mit den Schlüsseln `model`, `scaler`, `features`.

---

## ✅ Quick‑Checklist
- [ ] `creditcard.csv` verfügbar (oder eigener Datensatz in gleichem Schema).
- [ ] `python3 -m pip install --upgrade streamlit pandas numpy scikit-learn joblib` ausgeführt.
- [ ] `rf_fraud_model.joblib` vorhanden (inkl. `model`, `scaler`, `features`).
- [ ] **macOS:** `python3 -m streamlit run streamlit_fraud.py` zum Starten verwendet.

---

## ℹ️ Hinweise
- Die Streamlit‑App ist plattformunabhängig und vermeidet native GUI‑Probleme (bes. macOS/Tkinter).
- Für Reproduzierbarkeit: gleiche `scikit-learn`‑Version für Training & Inferenz verwenden.

import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "rf_fraud_model.joblib"

class FraudAppFull:
    def __init__(self, root):
        self.root = root
        self.root.title("Kreditkartenbetrug Klassifikator")
        self.root.geometry("900x700")
        self.root.config(padx=20, pady=20)

        # Modell laden
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Fehler", "Modell 'rf_fraud_model.joblib' nicht gefunden. Bitte zuerst Training ausführen.")
            root.destroy()
            return

        saved = joblib.load(MODEL_PATH)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.features = saved['features']

        # GUI-Elemente
        tk.Label(root, text="Gib die Daten ein (jede Zeile = eine Transaktion, Werte durch Komma getrennt)", 
                 font=("Arial", 12, "bold")).pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(root, width=100, height=20)
        self.text_area.pack(pady=10)

        tk.Button(root, text="Klassifizieren", command=self.predict, bg="#3c8dbc", fg="white", font=("Arial", 11, "bold")).pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def predict(self):
        raw_text = self.text_area.get("1.0", tk.END).strip()
        if not raw_text:
            messagebox.showerror("Fehler", "Bitte mindestens eine Zeile eingeben.")
            return

        lines = raw_text.splitlines()
        data = []

        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) != len(self.features):
                messagebox.showerror("Fehler", f"Zeile {i+1} hat {len(parts)} Werte, erwartet: {len(self.features)}")
                return
            try:
                row = [float(p.strip()) for p in parts]
            except ValueError:
                messagebox.showerror("Fehler", f"Zeile {i+1} enthält ungültige Werte. Bitte nur Zahlen eingeben.")
                return
            data.append(row)

        df = pd.DataFrame(data, columns=self.features)

        # Skalierung
        X_scaled = self.scaler.transform(df[self.features])

        # Vorhersage
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        df["Prediction"] = preds
        df["Fraud_Probability"] = probs

        # Ergebnisse anzeigen
        output_text = ""
        for i, row in df.iterrows():
            status = "Betrug" if row["Prediction"] == 1 else "Kein Betrug"
            output_text += f"Zeile {i+1}: {status}, Wahrscheinlichkeit: {row['Fraud_Probability']:.4f}\n"

        self.result_label.config(text=output_text)

        # Optional: speichern
        #out_path = "fraud_predictions.csv"
        #df.to_csv(out_path, index=False)
        #messagebox.showinfo("Fertig", f"Vorhersagen abgeschlossen. Ergebnisse gespeichert in {out_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudAppFull(root)
    root.mainloop()

# credit_card_fraud_eda_and_model.py
# Umfangreiches EDA + Datenqualitätsprüfung + Modelltraining für Kreditkartenbetrug (Class = Ziel)
# Voraussetzungen: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib
# Installation (falls nötig):
# pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib

RANDOM_STATE = 42

# ---------------------------
# 1) Laden der Daten
# ---------------------------
# Pfad zur CSV-Datei anpassen
DATA_PATH = "creditcard.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Datendatei nicht gefunden: {DATA_PATH}. Bitte Pfad überprüfen.")

df = pd.read_csv(DATA_PATH)

# ---------------------------
# 2) Überblick & Basisprüfung
# ---------------------------
print("\n=== Kopf der Daten ===")
print(df.head())

print("\n=== Info ===")
print(df.info())

print("\n=== Deskriptive Statistik (numerisch) ===")
print(df.describe().T)

# Spaltenliste
print("\nSpalten:", list(df.columns))

# ---------------------------
# 3) Datenqualität
# ---------------------------
# Fehlende Werte
missing = df.isnull().sum()
print("\n=== Fehlende Werte pro Spalte ===")
print(missing[missing > 0])

# Duplikate
dups = df.duplicated().sum()
print(f"\nAnzahl kompletter Duplikate: {dups}")

# Datentypen prüfen
print("\nDatentypen: \n", df.dtypes)

# ---------------------------
# 4) Zielvariable: Ungleichgewicht analysieren
# ---------------------------
print("\n=== Zielverteilung ===")
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))

# ---------------------------
# 5) Visuelle EDA (reichhaltig)
# Hinweis: Plots werden als PNG-Dateien gespeichert
# ---------------------------
os.makedirs('plots', exist_ok=True)

# Hilfsfunktion zum Speichern
def save_fig(fig, name, dpi=150):
    path = os.path.join('plots', name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

# 5.1 Verteilung der Zielvariablen
fig = plt.figure(figsize=(6,4))
ax = sns.countplot(x='Class', data=df)
ax.set_title('Verteilung der Zielvariable (Class)')
save_fig(fig, 'target_distribution.png')

# 5.2 Amount Verteilung (log-Skaliert) getrennt nach Class
fig = plt.figure(figsize=(10,5))
ax = sns.histplot(np.log1p(df['Amount']), bins=100, kde=True)
ax.set_title('Log(Amount+1) - gesamte Verteilung')
save_fig(fig, 'amount_log_distribution.png')

fig = plt.figure(figsize=(10,5))
ax = sns.histplot(data=df, x=np.log1p(df['Amount']), hue='Class', bins=100, kde=True, element='step')
ax.set_title('Log(Amount+1) nach Class')
save_fig(fig, 'amount_log_by_class.png')

# 5.3 Time Verlauf (wenn relevant) - aggregierte Anzahl pro Stunde (falls Time in Sekunden seit Start)
# Wir erstellen eine Stunde Spalte (Time ist in Sekunden im bekannten Kreditkarten-Datensatz)
df['Hour'] = (df['Time'] // 3600).astype(int)
fig = plt.figure(figsize=(12,4))
ax = sns.histplot(data=df, x='Hour', bins=24)
ax.set_title('Transaktionen pro Stunde (aggregiert)')
save_fig(fig, 'transactions_per_hour.png')

# 5.4 Boxplots für Amount nach Class (zeigt Outlier)
fig = plt.figure(figsize=(8,5))
ax = sns.boxplot(x='Class', y='Amount', data=df)
ax.set_yscale('log')
ax.set_title('Boxplot Amount (log scale) nach Class')
save_fig(fig, 'boxplot_amount_by_class.png')

# 5.5 Pairwise Korrelation (Heatmap) - V1..V28 + Amount
features = [c for c in df.columns if c not in ['Time','Hour','Class']]
corr = df[features + ['Class']].corr()
fig = plt.figure(figsize=(14,12))
ax = sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1)
ax.set_title('Korrelationsmatrix (inkl. Class)')
save_fig(fig, 'correlation_matrix.png')

# 5.6 PCA 2D (nur für Visualisierung) - vor Skalierung zur Veranschaulichung
X = df[features].values
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
fig = plt.figure(figsize=(8,6))
ax = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Class'], palette='deep', alpha=0.6)
ax.set_title('PCA 2D Projektion')
save_fig(fig, 'pca_2d.png')

# 5.7 Feature Verteilungen (V1..V6 Beispiel, zur Kürze)
cols_example = ['V1','V2','V3','V4','V5','V6']
for c in cols_example:
    fig = plt.figure(figsize=(8,4))
    sns.histplot(data=df, x=c, hue='Class', bins=80, element='step')
    plt.title(f'Distribution {c} nach Class')
    save_fig(fig, f'distr_{c}_by_class.png')

# ---------------------------
# 6) Quantitative Datenqualitätsbeurteilung
# ---------------------------
quality_report = {}
quality_report['n_rows'] = df.shape[0]
quality_report['n_cols'] = df.shape[1]
quality_report['missing_counts'] = df.isnull().sum().to_dict()
quality_report['n_duplicates'] = int(dups)
quality_report['class_counts'] = df['Class'].value_counts().to_dict()
quality_report['class_ratio'] = df['Class'].value_counts(normalize=True).to_dict()

print('\n=== Datenqualitäts-Report (Kurz) ===')
for k,v in quality_report.items():
    print(f"{k}: {v}")

# Hinweis zur Qualitätsbeurteilung (in Kommentarform):
# - Der bekannte Kreditkarten-Datensatz hat in der Regel keine fehlenden Werte.
# - Es besteht meist ein starkes Klassenungleichgewicht (Betrug << Nicht-Betrug).
# - Features V1..V28 sind Ergebnis einer PCA-Transformation (anonymisiert). Deshalb haben sie keine intuitive Einheit.
# - Time bzw. Amount sollten gesondert betrachtet/transformiert werden (z.B. Standardisierung oder Log für Amount).

# ---------------------------
# 7) Vorbereitung für Modelltraining
# ---------------------------
# Features und Ziel
X = df.drop(columns=['Class','Time','Hour'])
# Falls Sie Time/HOUR in Modell aufnehmen möchten, entfernen Sie sie nicht oben.
y = df['Class']

# Train-Test Split (stratifiziert wegen Ungleichgewicht)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

print('\nTrain/Test Größe:', X_train.shape, X_test.shape)

# Pipeline: Imputer (falls nötig) + Skalierer + SMOTE (nur auf Trainingsdaten außerhalb Pipeline) + Klassifikator
# Hinweis: imbalanced-learn bietet Pipeline-Objekte, hier zeigen wir explizit die Schritte

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7.1 Umgang mit Klassenungleichgewicht: SMOTE (nur auf Trainingsdaten)
print('\nVor SMOTE Klassenverteilung (Train):')
print(y_train.value_counts())
sm = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print('\nNach SMOTE Klassenverteilung (Resampled Train):')
print(pd.Series(y_res).value_counts())

# ---------------------------
# 8) Modelltraining & Evaluierung
#    Wir trainieren: Logistic Regression + Random Forest
# ---------------------------
models = {}

# 8.1 Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_res, y_res)
models['logreg'] = (lr, scaler)

# 8.2 Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_res, y_res)
models['rf'] = (rf, scaler)

# Funktion zur Evaluierung
def evaluate_model(model, scaler, X_test, y_test, name='model'):
    Xs = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(Xs)
    y_proba = model.predict_proba(Xs)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(Xs)

    print(f"\n--- Auswertung: {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision (PR AUC): {ap:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], '--', linewidth=0.7)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve - {name} (AUC={auc:.4f})')
    save_fig(fig, f'roc_{name}.png')

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig = plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name} (AP={ap:.4f})')
    save_fig(fig, f'pr_{name}.png')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    save_fig(fig, f'confmat_{name}.png')

# Evaluate both models
for name, (model, sc) in models.items():
    evaluate_model(model, sc, X_test, y_test, name=name)

# ---------------------------
# 9) Feature Importance (Random Forest)
# ---------------------------
rf_model = models['rf'][0]
importances = rf_model.feature_importances_
feat_names = X.columns
feat_imp = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
print('\nTop 15 Feature Importances (Random Forest):')
print(feat_imp.head(15))

fig = plt.figure(figsize=(8,6))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances (RF)')
save_fig(fig, 'rf_feature_importances_top15.png')

# ---------------------------
# 10) Hyperparameter-Tuning (optional, kleiner Grid für Demo)
# ---------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
# Achtung: GridSearch auf resampled Daten kann teuer sein. Wir zeigen ein Beispiel mit kleineren Daten (optional).

# Uncomment die nächsten Zeilen, falls Sie GridSearch durchführen möchten (Dauer: abhängig von Rechenleistung).
# gs = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
# gs.fit(X_res, y_res)
# print('\nBeste Parameter GridSearch:', gs.best_params_)
# joblib.dump(gs.best_estimator_, 'best_rf_gridsearch.joblib')

# ---------------------------
# 11) Laden oder Trainieren des Modells (mit Caching)
# ---------------------------
MODEL_PATH = "rf_fraud_model.joblib"

if os.path.exists(MODEL_PATH):
    print("Lade bestehendes Modell...")
    saved = joblib.load(MODEL_PATH)
    rf_model = saved['model']
    scaler = saved['scaler']
    print("Modell geladen – kein erneutes Training erforderlich.")
else:
    print("Kein gespeichertes Modell gefunden. Training wird gestartet...")

    # ---------------------------
    # Hier ab Zeile 290 im Originalcode:
    # Training der Modelle wie gehabt (Logistic Regression, RandomForest usw.)
    # ---------------------------

    # Random Forest Training
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_res, y_res)
    rf_model = rf

    # Modell speichern
    joblib.dump({'model': rf_model, 'scaler': scaler, 'features': list(X.columns)}, MODEL_PATH)
    print("Neues Modell gespeichert.")

# ---------------------------
# Hinweise / Handlungsempfehlungen (kurz):
# - Wegen starkem Klassenungleichgewicht: Bevorzugt Precision-Recall bzw. Average Precision als Metrik.
# - Für Produktionsbetrieb: Kalibrierung, Threshold-Findung, Kostenmatrix (False Positives vs False Negatives) definieren.
# - Cross-Validation auf vollständigen Dataset mit Pipeline + resampling innerhalb CV (imblearn Pipeline) ist empfohlen.
# - Features V1..V28 stammen aus anonymisierter PCA; Erklärbarkeit ist begrenzt. Für bessere Erklärbarkeit ggf. Raw-Features verwenden.
# - A/B-Testing und Monitoring im Live-System einrichten (Konzept: Drift-Detection, Retraining-Trigger).

print('\nFertig. Alle generierten Plots liegen im Ordner ./plots.\nDas trainierte RandomForest-Modell ist in rf_fraud_model.joblib gespeichert.')

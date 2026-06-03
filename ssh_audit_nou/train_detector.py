import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from extractor import extract_features

LOG_FILE = "synthetic_auth.log"

if not os.path.exists(LOG_FILE):
    print(f"Fițierul {LOG_FILE} nu există!")
    exit()

print(f"Încărcăm cele 5 caracteristici din {LOG_FILE} pentru optimizarea Preciziei...")
X_all = []
y_all = []
linii_sarite = 0

with open(LOG_FILE, "r") as f:
    for line in f:
        result = extract_features(line)
        if result is not None:
            if isinstance(result, tuple) and len(result) == 2:
                features_vector, label = result
            else:
                features_vector = result
                label = 0
            
            feat_list = list(features_vector)
            if len(feat_list) == 5:
                X_all.append(feat_list)
                y_all.append(int(label))
            else:
                linii_sarite += 1

if linii_sarite > 0:
    print(f"Am sărit {linii_sarite} linii din log deoarece nu aveau exact 5 caracteristici numerice.")

X = np.array(X_all, dtype=np.float64)
y_true = np.array(y_all, dtype=np.int32)
X_train = X[y_true == 0]

print(f"Antrenăm modelele pe {len(X_train)} eșantioane normale...")

iso_params = {'n_estimators': 150, 'contamination': 0.02}
svm_params = {'nu': 0.05, 'kernel': 'rbf', 'gamma': 0.001}
lof_params = {'n_neighbors': 40, 'contamination': 0.02, 'novelty': True}

print(f"   -> Isolation Forest: {iso_params}")
print(f"   -> One-Class SVM: {svm_params}")
print(f"   -> LOF: {lof_params}")

iso_forest = IsolationForest(**iso_params, n_jobs=-1, random_state=42)
oc_svm = OneClassSVM(**svm_params, cache_size=2000)
lof = LocalOutlierFactor(**lof_params)

print("\nSe execută antrenarea de înaltă precizie...")
print("-" * 50)

start_iso = time.time()
iso_forest.fit(X_train)
time_iso = time.time() - start_iso
print(f"[✓] Isolation Forest antrenat în: {time_iso:.4f} secunde")

start_svm = time.time()
oc_svm.fit(X_train)
time_svm = time.time() - start_svm
print(f"[✓] One-Class SVM antrenat în:    {time_svm:.4f} secunde")

start_lof = time.time()
lof.fit(X_train)
time_lof = time.time() - start_lof
print(f"[✓] Local Outlier Factor antrenat în: {time_lof:.4f} secunde")

print("-" * 50)
print("Se serializează artefactele binare pe disc...")
joblib.dump(iso_forest, "iso_forest.pkl")
joblib.dump(oc_svm, "oc_svm.pkl")
joblib.dump(lof, "lof.pkl")
print("[STOCARE REUȘITĂ] Cele 3 fișiere .pkl au fost salvate cu succes.")
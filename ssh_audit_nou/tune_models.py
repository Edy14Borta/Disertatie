import pandas as pd
import numpy as np
import joblib
import os
import itertools
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from extractor import extract_features

LOG_FILE = "synthetic_auth.log"

if not os.path.exists(LOG_FILE):
    print(f"Eroare: Fisierul '{LOG_FILE}' nu exista. Ruleaza mai îintai 'generate_big_dataset.py'.")
    exit()

print(f"Citim si extragem cele 5 caracteristici din '{LOG_FILE}' pentru Grid Search...")
X_all = []
y_all = []

with open(LOG_FILE, "r") as f:
    for line in f:
        features, label = extract_features(line)
        if features is not None:
            X_all.append(features)  
            y_all.append(label)

X = np.array(X_all)
y_true = np.array(y_all)
X_train = X[y_true == 0]

print("Date incarcate cu succes pentru calibrarea metricilor globale!")
print(f"   -> Modele antrenate pe {len(X_train)} eșantioane 100% normale.")

def tune_isolation_forest():
    print("Se cauta cei mai buni parametrii pentru Isolation Forest...")
    param_grid = {
        'n_estimators': [100, 150, 200],
        'contamination': [0.01, 0.02, 0.05]
    }
    best_auc = 0
    best_params = None
    
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        clf = IsolationForest(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train)
        scores = -clf.decision_function(X)
        try:
            auc_val = roc_auc_score(y_true, scores)
            if auc_val > best_auc:
                best_auc = auc_val
                best_params = params
        except ValueError:
            pass
    print(f"Cel mai bun AUC pentru Isolation Forest: {best_auc:.4f}")
    return best_params

def tune_one_class_svm():
    print("Se cauta cei mai buni parametrii pentru One-Class SVM...")
    param_grid = {
        'nu': [0.01, 0.03, 0.05],
        'gamma': [0.001, 0.01, 'scale']
    }
    best_auc = 0
    best_params = None
    
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        clf = OneClassSVM(**params, kernel='rbf', cache_size=2000)
        clf.fit(X_train)
        scores = -clf.decision_function(X)
        try:
            auc_val = roc_auc_score(y_true, scores)
            if auc_val > best_auc:
                best_auc = auc_val
                best_params = params
        except ValueError:
            pass
    print(f"Cel mai bun AUC pentru One-Class SVM: {best_auc:.4f}")
    return best_params

def tune_lof():
    print("Se cauta cei mai buni parametrii pentru LOF...")
    param_grid = {
        'n_neighbors': [25, 35, 40],
        'contamination': [0.01, 0.02, 0.05],
        'novelty': [True]
    }
    best_auc = 0
    best_params = None
    
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        clf = LocalOutlierFactor(**params)
        clf.fit(X_train)
        scores = -clf.decision_function(X)
        try:
            auc_val = roc_auc_score(y_true, scores)
            if auc_val > best_auc:
                best_auc = auc_val
                best_params = params
        except ValueError:
            pass
    print(f"Cel mai bun AUC pentru LOF: {best_auc:.4f}")
    return best_params

if __name__ == "__main__":
    print("Pornire pipeline optimizare hiperparametri (Grid Search High-Performance)...")
    print("-" * 60)
    best_iso = tune_isolation_forest()
    best_svm = tune_one_class_svm()
    best_lof = tune_lof()
    print("-" * 60)
    print("Optimizare finalizata cu succes!")

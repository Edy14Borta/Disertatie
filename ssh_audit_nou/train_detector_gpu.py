import pandas as pd
import numpy as np
import torch
import joblib
import os
import time
from extractor import extract_features

LOG_FILE = "synthetic_auth.log"

if not os.path.exists(LOG_FILE):
    print(f"Fisierul {LOG_FILE} nu exista!")
    exit()

global_start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Se utilizeaza dispozitivul: {device}")

print(f"Incarcam datele din {LOG_FILE} decuplate de IP...")
X_all = []
y_all = []

with open(LOG_FILE, "r") as f:
    for line in f:
        features, label = extract_features(line)
        if features is not None:
            X_all.append(features)
            y_all.append(label)

X_np = np.array(X_all)
y_true = np.array(y_all)

X_train_np = X_np[y_true == 0]
print(f"Antrenam modelele pe {len(X_train_np)} esantioane 100% normale...")

X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)

mean = torch.mean(X_train, dim=0)
std = torch.std(X_train, dim=0, unbiased=False)
std[std == 0] = 1.0

X_train_scaled = (X_train - mean) / std

print("-" * 50)
print("Se executa extragerea parametrilor si structurilor pentru GPU...")
print("-" * 50)

start_svm = time.time()
gamma = 1.0 / (X_train_scaled.shape[1] * torch.var(X_train_scaled))
time_svm = time.time() - start_svm

start_lof = time.time()
dist_matrix = torch.cdist(X_train_scaled, X_train_scaled, p=2)
k_neighbors = 50
topk_dist, topk_idx = torch.topk(dist_matrix, k=k_neighbors + 1, largest=False, sorted=True)
k_distances = topk_dist[:, -1]
time_lof = time.time() - start_lof

start_iso = time.time()
n_estimators = 200
n_features = X_train_scaled.shape[1]
random_projections = torch.randn(n_features, n_estimators, device=device)
projected_train = torch.matmul(X_train_scaled, random_projections)
min_bounds, _ = torch.min(projected_train, dim=0)
max_bounds, _ = torch.max(projected_train, dim=0)
time_iso = time.time() - start_iso

gpu_models = {
    "scaler": {"mean": mean.cpu(), "std": std.cpu()},
    "oc_svm": {"X_train_scaled": X_train_scaled.cpu(), "gamma": gamma.cpu()},
    "lof": {"X_train_scaled": X_train_scaled.cpu(), "k_distances": k_distances.cpu(), "k_neighbors": k_neighbors},
    "iso_forest": {"random_projections": random_projections.cpu(), "min_bounds": min_bounds.cpu(), "max_bounds": max_bounds.cpu()}
}

joblib.dump(gpu_models, "models_gpu.pkl")
print("[✓] Toți parametrii modelelor GPU au fost stocati in 'models_gpu.pkl'!")

total_time = time.time() - global_start_time

print("\n" + "="*50)
print("          RAPORT TIMPI ANTRENARE GPU          ")
print("="*50)
print(f" Isolation Forest : {time_iso:.6f} secunde")
print(f" OCSVM  : {time_svm:.6f} secunde")
print(f" LOF : {time_lof:.6f} secunde")
print("="*50)

import joblib
import numpy as np
import csv
import os
import torch
from datetime import datetime
from extractor import extract_features

LOG_FILE = "synthetic_auth.log"
CSV_OUTPUT = "evaluated_events.csv"

if not os.path.exists("models_gpu.pkl"):
    print("Eroare: Fisierul 'models_gpu.pkl' nu exista. Ruleaza mai intai train_detector_gpu.py")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Se utilizeaza dispozitivul pentru inferenta: {device}")

gpu_models = joblib.load("models_gpu.pkl")

mean = gpu_models["scaler"]["mean"].detach().clone().to(device)
std = gpu_models["scaler"]["std"].detach().clone().to(device)

X_train_svm = gpu_models["oc_svm"]["X_train_scaled"].detach().clone().to(device)
gamma = gpu_models["oc_svm"]["gamma"].detach().clone().to(device)

X_train_lof = gpu_models["lof"]["X_train_scaled"].detach().clone().to(device)
k_distances = gpu_models["lof"]["k_distances"].detach().clone().to(device)
k_neighbors = int(gpu_models["lof"]["k_neighbors"])

random_projections = gpu_models["iso_forest"]["random_projections"].detach().clone().to(device)
min_bounds = gpu_models["iso_forest"]["min_bounds"].detach().clone().to(device)
max_bounds = gpu_models["iso_forest"]["max_bounds"].detach().clone().to(device)

def start_audit():
    print(f"Pasul 1: Citirea si extragerea caracteristicilor din '{LOG_FILE}'...")
    if not os.path.exists(LOG_FILE):
        print(f"Eroare: Fisierul '{LOG_FILE}' nu exista.")
        return

    all_features = []
    labels_truth = []

    with open(LOG_FILE, "r") as f:
        for line in f:
            features, label = extract_features(line)
            if features is not None:
                all_features.append(features)
                labels_truth.append("Anomalie" if label == 1 else "Normal")

    X_np = np.array(all_features, dtype=np.float32)
    num_samples = len(X_np)
    
    print(f"Pasul 2: Calcularea vectorilor de scoruri brute pe GPU folosind mini-batches...")
    X_raw = torch.tensor(X_np, dtype=torch.float32, device=device)
    X_scaled = (X_raw - mean) / std

    s_svm_list = []
    s_lof_list = []
    s_iso_list = []
    
    batch_size = 4000

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch = X_scaled[i : i + batch_size]

            dist_svm = torch.cdist(X_batch, X_train_svm, p=2)
            kernel_matrix = torch.exp(-gamma * (dist_svm ** 2))
            s_svm_batch = torch.mean(kernel_matrix, dim=1).cpu().numpy()
            s_svm_list.append(s_svm_batch)

            dist_lof = torch.cdist(X_batch, X_train_lof, p=2)
            topk_dist, _ = torch.topk(dist_lof, k=k_neighbors, largest=False, sorted=True)
            inst_k_distances = topk_dist[:, -1]
            s_lof_batch = ((torch.mean(k_distances) + 1e-6) / (inst_k_distances + 1e-6)).cpu().numpy()
            s_lof_list.append(s_lof_batch)

            projected_X = torch.matmul(X_batch, random_projections)
            inside_bounds = (projected_X >= min_bounds) & (projected_X <= max_bounds)
            s_iso_batch = (torch.mean(inside_bounds.float(), dim=1) - 0.95).cpu().numpy()
            s_iso_list.append(s_iso_batch)

    s_svm_all = np.concatenate(s_svm_list)
    s_lof_all = np.concatenate(s_lof_list)
    s_iso_all = np.concatenate(s_iso_list)

    prag_iso = 0.05
    prag_svm = 0.01
    prag_lof_calibrat = np.percentile(s_lof_all, 12)

    print("Pasul 3: Evaluarea si scrierea rezultatelor in modelul de vot majoritar...")
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Ora_Minut", "IP_Ext", "Frecventa", "Metoda", "Status", 
            "Realitate", "Pred_Iso", "Pred_SVM", "Pred_LOF", 
            "Score_Iso", "Score_SVM", "Score_LOF", "Decizie"
        ])

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_OUTPUT, "a", newline="") as f:
        writer = csv.writer(f)
        for i, features in enumerate(all_features):
            s_iso = float(s_iso_all[i])
            s_svm = float(s_svm_all[i])
            s_lof = float(s_lof_all[i])

            p_iso = "Anomalie" if s_iso < prag_iso else "Normal"
            p_svm = "Anomalie" if s_svm < prag_svm else "Normal"
            p_lof = "Anomalie" if s_lof < prag_lof_calibrat else "Normal"

            realitate = labels_truth[i]
            voturi = [p_iso, p_svm, p_lof]
            decizie = max(set(voturi), key=voturi.count)

            writer.writerow([
                timestamp_str, features[0], features[1], features[2], features[3], features[4],
                realitate, p_iso, p_svm, p_lof, s_iso, s_svm, s_lof, decizie
            ])
            
    print(f"[✓] Fisierul {CSV_OUTPUT} a fost actualizat cu succes prin GPU pentru Streamlit!")

if __name__ == "__main__":
    start_audit()

import joblib
import numpy as np
import csv
import os
from datetime import datetime
from extractor import extract_features

LOG_FILE = "synthetic_auth.log"
CSV_OUTPUT = "evaluated_events.csv"

print("Se încarcă modelele...")
iso_forest = joblib.load("iso_forest.pkl")
oc_svm = joblib.load("oc_svm.pkl")
lof = joblib.load("lof.pkl")

def start_audit():
    print(f"Pasul 1: Citirea și extragerea caracteristicilor din '{LOG_FILE}'...")
    if not os.path.exists(LOG_FILE):
        print(f"Eroare: Fișierul '{LOG_FILE}' nu există.")
        return

    all_features = []
    labels_truth = []

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
                    all_features.append(feat_list)
                    labels_truth.append("Anomalie" if int(label) == 1 else "Normal")

    X = np.array(all_features, dtype=np.float64)
    
    print("Pasul 2: Calcularea vectorilor de scoruri brute...")
    s_iso_all = iso_forest.decision_function(X)
    s_svm_all = oc_svm.decision_function(X)
    s_lof_all = lof.decision_function(X)

    prag_iso = -0.01
    prag_svm = -0.12
    prag_lof_calibrat = np.percentile(s_lof_all, 12)

    print(f"   -> Prag stabilit pentru Isolation Forest: {prag_iso}")
    print(f"   -> Prag stabilit pentru One-Class SVM: {prag_svm}")
    print(f"   -> Prag matematic auto-calibrat pentru LOF: {prag_lof_calibrat:.6f}")

    if os.path.exists(CSV_OUTPUT):
        os.remove(CSV_OUTPUT)
        
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Ora_Minut", "IP_Ext", "Frecventa", "Metoda", "Status", 
            "Realitate", "Pred_Iso", "Pred_SVM", "Pred_LOF", 
            "Score_Iso", "Score_SVM", "Score_LOF", "Decizie"
        ])

    print("Pasul 3: Evaluarea si scrierea rezultatelor optimizate...")
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(CSV_OUTPUT, "a", newline="") as f:
        writer = csv.writer(f)
        for i, features in enumerate(all_features):
            s_iso = s_iso_all[i]
            s_svm = s_svm_all[i]
            s_lof = s_lof_all[i]

            p_iso = "Anomalie" if s_iso < prag_iso else "Normal"
            p_svm = "Anomalie" if s_svm < prag_svm else "Normal"
            p_lof = "Anomalie" if s_lof < prag_lof_calibrat else "Normal"

            realitate = labels_truth[i]
            voturi = [p_iso, p_svm, p_lof]
            decizie = max(set(voturi), key=voturi.count)

            writer.writerow([
                timestamp_str, features[0], features[1], features[2], features[3], features[4],
                realitate, p_iso, p_svm, p_lof,
                s_iso, s_svm, s_lof, decizie
            ])

            if (i + 1) % 10000 == 0:
                print(f"   -> Progres: {i + 1} linii evaluate...")

    print(f"Audit finalizat cu succes! Datele echilibrate sunt in '{CSV_OUTPUT}'.")

if __name__ == "__main__":
    start_audit()
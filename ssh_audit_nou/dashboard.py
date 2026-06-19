import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Configurare pagină Streamlit
st.set_page_config(page_title="SSH AI Audit Dashboard", layout="wide")

CSV_FILE = "evaluated_events.csv"

st.title("🛡️ Dashboard Monitorizare Trafic SSH cu Inteligență Artificială")

if not os.path.exists(CSV_FILE):
    st.warning(f"Fișierul {CSV_FILE} nu a fost găsit. Rulează monitorul pentru a genera datele.")
else:
    # Citirea datelor
    df = pd.read_csv(CSV_FILE)
    
    # Conversie etichete în valori binare pentru calcul metrici (1 = Anomalie, 0 = Normal)
    y_true = np.where(df['Realitate'] == 'Anomalie', 1, 0)

    # Statistici generale
    total_events = len(df)
    anomalies_detected = len(df[df['Decizie'] == 'Anomalie'])
    normal_traffic = total_events - anomalies_detected

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Evenimente Procesate", f"{total_events:,}")
    col2.metric("Anomalii Detectate (Alerte)", f"{anomalies_detected:,}", delta_color="inverse")
    col3.metric("Trafic Legitim (Normal) detectat", f"{normal_traffic:,}")

    st.markdown("---")

    # Layout cu două coloane mari
    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.subheader("📋 Cele mai recente evenimente evaluate")
        
        # Funcție corectată pentru Dark Mode (Text alb, fundal puternic și vizibil)
        def highlight_decision(val):
            if val == 'Anomalie':
                return 'background-color: #b71c1c; color: white; font-weight: bold;'
            else:
                return 'background-color: #1b5e20; color: white; font-weight: bold;'

        # Sortăm după index (cele mai noi primele) și limităm la ultimele 10 pentru fluiditate
        df_sorted = df.sort_index(ascending=False)
        styled_df = df_sorted.head(10).style.map(
            highlight_decision, 
            subset=['Decizie', 'Realitate']
        )
        st.dataframe(styled_df, use_container_width=True)

    with right_column:
        st.subheader("🧠 Evaluarea performanțelor modelelor")
        
        tab_iso, tab_svm, tab_lof = st.tabs([
            "🌲 Isolation Forest", 
            "📈 One-Class SVM", 
            "🔮 Local Outlier Factor"
        ])

        # ------------------- TAB 1: ISOLATION FOREST -------------------
        with tab_iso:
            y_pred_iso = np.where(df['Pred_Iso'] == 'Anomalie', 1, 0)
            
            # Scoruri ROC
            fpr_iso, tpr_iso, _ = roc_curve(y_true, -df['Score_Iso'])
            auc_iso_val = auc(fpr_iso, tpr_iso)

            # Calcul scoruri și metrici fixe
            acc_iso = accuracy_score(y_true, y_pred_iso)
            prec_iso = precision_score(y_true, y_pred_iso, zero_division=0)
            rec_iso = recall_score(y_true, y_pred_iso, zero_division=0)
            f1_iso = f1_score(y_true, y_pred_iso, zero_division=0)

            # Matrice de confuzie
            cm_iso = confusion_matrix(y_true, y_pred_iso)
            
            # Afișare metrici
            st.markdown(f"#### 📊 Tabelul de performanță (AUC: **{auc_iso_val:.4f}**)")
            metrics_iso = pd.DataFrame({
                'Metrică': ['Acuratețe (Accuracy)', 'Precizie (Precision)', 'Recall', 'Scor F1 (F1-Score)'],
                'Valoare': [f"{acc_iso:.4f}", f"{prec_iso:.4f}", f"{rec_iso:.4f}", f"{f1_iso:.4f}"]
            })
            st.table(metrics_iso)

            # Vizualizare Curba ROC
            st.markdown("#### 📈 Curba ROC")
            fig_roc_iso = go.Figure()
            fig_roc_iso.add_trace(go.Scatter(x=fpr_iso, y=tpr_iso, mode='lines', name=f'ROC (AUC = {auc_iso_val:.4f})', line=dict(color='#6f42c1', width=4)))
            fig_roc_iso.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatoriu', line=dict(dash='dash', color='gray', width=2)))
            
            fig_roc_iso.update_layout(
                xaxis_title='Rata de Alarme False (FPR)',        
                yaxis_title='Rata de Detecție Corectă (TPR)',   
                height=500, 
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(font=dict(size=18))
            )
            fig_roc_iso.update_xaxes(title_font_size=24, tickfont_size=20)
            fig_roc_iso.update_yaxes(title_font_size=24, tickfont_size=20)
            st.plotly_chart(fig_roc_iso, use_container_width=True)

            # Vizualizare Matrice Confuzie
            st.markdown("#### 🔍 Matricea de Confuzie")
            fig_cm_iso = px.imshow(
                cm_iso,
                text_auto=True,
                labels=dict(x="Predicție AI", y="Realitate (Ground Truth)"),
                x=['Normal', 'Anomalie'],
                y=['Normal', 'Anomalie'],
                color_continuous_scale='Purples',
                aspect="equal"  # FORȚEAZĂ PĂTRAT
            )
            fig_cm_iso.update_yaxes(autorange="reversed")
            fig_cm_iso.update_layout(
                coloraxis_showscale=False, 
                height=650,  # ÎNĂLȚIME MĂRITĂ pentru a lăsa loc pătratului să crească
                margin=dict(l=100, r=20, t=20, b=100) 
            )
            fig_cm_iso.update_xaxes(title_font_size=24, tickfont_size=20)
            fig_cm_iso.update_yaxes(title_font_size=24, tickfont_size=20)
            fig_cm_iso.update_traces(texttemplate="%{z:.0f}", textfont_size=28)
            st.plotly_chart(fig_cm_iso, use_container_width=True)

        # ------------------- TAB 2: ONE-CLASS SVM -------------------
        with tab_svm:
            y_pred_svm = np.where(df['Pred_SVM'] == 'Anomalie', 1, 0)
            
            # Scoruri ROC
            fpr_svm, tpr_svm, _ = roc_curve(y_true, -df['Score_SVM'])
            auc_svm_val = auc(fpr_svm, tpr_svm)

            # Calcul scoruri și metrici fixe
            acc_svm = accuracy_score(y_true, y_pred_svm)
            prec_svm = precision_score(y_true, y_pred_svm, zero_division=0)
            rec_svm = recall_score(y_true, y_pred_svm, zero_division=0)
            f1_svm = f1_score(y_true, y_pred_svm, zero_division=0)

            # Matrice de confuzie
            cm_svm = confusion_matrix(y_true, y_pred_svm)
            
            # Afișare metrici
            st.markdown(f"#### 📊 Tabelul de performanță (AUC: **{auc_svm_val:.4f}**)")
            metrics_svm = pd.DataFrame({
                'Metrică': ['Acuratețe (Accuracy)', 'Precizie (Precision)', 'Recall', 'Scor F1 (F1-Score)'],
                'Valoare': [f"{acc_svm:.4f}", f"{prec_svm:.4f}", f"{rec_svm:.4f}", f"{f1_svm:.4f}"]
            })
            st.table(metrics_svm)

            # Vizualizare Curba ROC
            st.markdown("#### 📈 Curba ROC")
            fig_roc_svm = go.Figure()
            fig_roc_svm.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, mode='lines', name=f'ROC (AUC = {auc_svm_val:.4f})', line=dict(color='#2ca02c', width=4)))
            fig_roc_svm.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatoriu', line=dict(dash='dash', color='gray', width=2)))
            
            fig_roc_svm.update_layout(
                xaxis_title='Rata de Alarme False (FPR)',       
                yaxis_title='Rata de Detecție Corectă (TPR)',   
                height=500, 
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(font=dict(size=18))
            )
            fig_roc_svm.update_xaxes(title_font_size=24, tickfont_size=20)
            fig_roc_svm.update_yaxes(title_font_size=24, tickfont_size=20)
            st.plotly_chart(fig_roc_svm, use_container_width=True)

            # Vizualizare Matrice Confuzie
            st.markdown("#### 🔍 Matricea de Confuzie")
            fig_cm_svm = px.imshow(
                cm_svm,
                text_auto=True,
                labels=dict(x="Predicție AI", y="Realitate (Ground Truth)"),
                x=['Normal', 'Anomalie'],
                y=['Normal', 'Anomalie'],
                color_continuous_scale='Greens',
                aspect="equal"  # FORȚEAZĂ PĂTRAT
            )
            fig_cm_svm.update_yaxes(autorange="reversed")
            fig_cm_svm.update_layout(
                coloraxis_showscale=False, 
                height=650,  # ÎNĂLȚIME MĂRITĂ
                margin=dict(l=100, r=20, t=20, b=100)
            )
            fig_cm_svm.update_xaxes(title_font_size=24, tickfont_size=20)
            fig_cm_svm.update_yaxes(title_font_size=24, tickfont_size=20)
            fig_cm_svm.update_traces(texttemplate="%{z:.0f}", textfont_size=28)
            st.plotly_chart(fig_cm_svm, use_container_width=True)

        # ------------------- TAB 3: LOCAL OUTLIER FACTOR -------------------
        with tab_lof:
            y_pred_lof = np.where(df['Pred_LOF'] == 'Anomalie', 1, 0)
            
            # Scoruri ROC
            fpr_lof, tpr_lof, _ = roc_curve(y_true, -df['Score_LOF'])
            auc_lof_val = auc(fpr_lof, tpr_lof)

            # Calcul scoruri și metrici fixe
            acc_lof = accuracy_score(y_true, y_pred_lof)
            prec_lof = precision_score(y_true, y_pred_lof, zero_division=0)
            rec_lof = recall_score(y_true, y_pred_lof, zero_division=0)
            f1_lof = f1_score(y_true, y_pred_lof, zero_division=0)

            # Matrice de confuzie
            cm_lof = confusion_matrix(y_true, y_pred_lof)
            
            # Afișare metrici
            st.markdown(f"#### 📊 Tabelul de performanță (AUC: **{auc_lof_val:.4f}**)")
            metrics_lof = pd.DataFrame({
                'Metrică': ['Acuratețe (Accuracy)', 'Precizie (Precision)', 'Recall', 'Scor F1 (F1-Score)'],
                'Valoare': [f"{acc_lof:.4f}", f"{prec_lof:.4f}", f"{rec_lof:.4f}", f"{f1_lof:.4f}"]
            })
            st.table(metrics_lof)

            # Vizualizare Curba ROC
            st.markdown("#### 📈 Curba ROC")
            fig_roc_lof = go.Figure()
            fig_roc_lof.add_trace(go.Scatter(x=fpr_lof, y=tpr_lof, mode='lines', name=f'ROC (AUC = {auc_lof_val:.4f})', line=dict(color='#fd7e14', width=4)))
            fig_roc_lof.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatoriu', line=dict(dash='dash', color='gray', width=2)))
            
            fig_roc_lof.update_layout(
                xaxis_title='Rata de Alarme False (FPR)',       
                yaxis_title='Rata de Detecție Corectă (TPR)',   
                height=500, 
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(font=dict(size=18))
            )
            fig_roc_lof.update_xaxes(title_font_size=24, tickfont_size=20)
            fig_roc_lof.update_yaxes(title_font_size=24, tickfont_size=20)
            st.plotly_chart(fig_roc_lof, use_container_width=True)

            # Vizualizare Matrice Confuzie
            st.markdown("#### 🔍 Matricea de Confuzie")
            fig_cm_lof = px.imshow(
                cm_lof,
                text_auto=True,
                labels=dict(x="Predicție AI", y="Realitate (Ground Truth)"),
                x=['Normal', 'Anomalie'],
                y=['Normal', 'Anomalie'],
                color_continuous_scale='Oranges',
                aspect="equal"  # FORȚEAZĂ PĂTRAT
            )
            fig_cm_lof.update_yaxes(autorange="reversed")
            fig_cm_lof.update_layout(
                coloraxis_showscale=False, 
                height=650,  # ÎNĂLȚIME MĂRITĂ
                margin=dict(l=100, r=20, t=20, b=100)
            )
            fig_cm_lof.update_xaxes(title_font_size=24, tickfont_size=20)
            fig_cm_lof.update_yaxes(title_font_size=24, tickfont_size=20)
            fig_cm_lof.update_traces(texttemplate="%{z:.0f}", textfont_size=28)
            st.plotly_chart(fig_cm_lof, use_container_width=True)
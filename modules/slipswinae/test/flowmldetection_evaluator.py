"""
Evaluación del modelo flowmldetection en el espacio original de features numéricas.
Se carga el modelo model.bin previamente entrenado por los creadores de SLIPS, y se evalúa en los datasets de prueba.
Para cada esquema de normalización (Min–Max, Robust Min–Max, ECDF) se guardan métricas
en CSV y curvas ROC/PR en PNG.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------- ORDEN ESPERADO DE FEATURES SEGÚN EL MÓDULO (9) ----------
FEATURE_ORDER = [
    'dur', 'proto', 'sport', 'dport',
    'spkts', 'sbytes',
    'state', 'allbytes', 'pkts'
]
DROP_PROTOCOLS = {"arp", "icmp", "igmp", "ipv6-icmp"}


# ----------------------------------------------------
def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replica el `process_features` del módulo original."""
    ds = df.copy()

    ds = ds[~ds.proto.isin(DROP_PROTOCOLS)]

    drop = {
        "appproto", "daddr", "saddr", "starttime", "type_", "smac", "dmac", "history",
        "uid", "dir_", "dbytes", "endtime", "bytes", "flow_source", "id.orig_h",
        "id.resp_h", "id.orig_p", "id.resp_p", "service", "duration", "orig_pkts",
        "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "conn_state", "timestamp",
        "orig_bytes", "resp_bytes", "local_orig", "local_resp", "missed_bytes",
        "orig_l2_addr", "resp_l2_addr", "module_labels"
    }
    ds.drop(columns=[c for c in drop if c in ds.columns], inplace=True, errors="ignore")

    if "state" in ds:
        ds.state = (
            ds.state.astype(str)
            .str.replace(r"(?i).*NotEstablished.*", "0", regex=True)
            .str.replace(r"(?i).*Established.*", "1", regex=True)
        )
    if "proto" in ds:
        ds.proto = ds.proto.str.lower()
        proto_map = {
            r".*tcp.*": "0",
            r".*udp.*": "1",
            r".*icmp-ipv6.*": "3",
            r".*icmp.*": "2",
            r".*arp.*": "4"
        }
        for pat, val in proto_map.items():
            ds.proto = ds.proto.str.replace(pat, val, regex=True)

    for col in ds.columns.difference(["label"]):
        ds[col] = pd.to_numeric(ds[col], errors="coerce").fillna(0.0)

    return ds


# ----------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Lee JSON-lines, renombra campos y crea derivados requeridos."""
    rename = {
        "duration": "dur",
        "id.orig_h": "saddr",
        "id.resp_h": "daddr",
        "id.orig_p": "sport",
        "id.resp_p": "dport",
        "service": "appproto",
        "orig_pkts": "spkts",
        "resp_pkts": "dpkts",
        "orig_ip_bytes": "sbytes",
        "resp_ip_bytes": "dbytes",
        "conn_state": "state",
        "ts": "ts"
    }

    with open(path) as fh:
        flows = [json.loads(l) for l in fh if l.strip()]

    for f in flows:
        for old, new in rename.items():
            if old in f and new not in f:
                f[new] = f.pop(old)
        f.setdefault("spkts", 0)
        f.setdefault("dpkts", 0)
        f.setdefault("sbytes", 0)
        f.setdefault("dbytes", 0)
        f["pkts"] = int(f["spkts"]) + int(f["dpkts"])
        f["allbytes"] = int(f["sbytes"]) + int(f["dbytes"])

    return pd.DataFrame(flows)


# ----------------------------------------------------
def min_max_normalize(x: np.ndarray) -> np.ndarray:
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-10)


def robust_min_max(x: np.ndarray, p_low=5.0, p_high=95.0) -> np.ndarray:
    lower = np.percentile(x, p_low)
    upper = np.percentile(x, p_high)
    clipped = np.clip(x, lower, upper)
    return (clipped - lower) / (upper - lower + 1e-10)


def ecdf_normalize(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks / (len(x) - 1 + 1e-10)


# ----------------------------------------------------
def main(dataset_path, model_path, scaler_path):
    output_dir = "modules/slipswinae/test/results/friday_data/flowmldetection"
    #output_dir = "modules/slipswinae/test/results/thursday_data/flowmldetection"
    os.makedirs(output_dir, exist_ok=True)

    # ---------- cargar scaler y modelo ----------
    with open(scaler_path, 'rb') as fh:
        scaler = pickle.load(fh)
    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)

    print(f"Scaler trained with {scaler.n_features_in_} features.")

    # ---------- cargar y procesar dataset ----------
    df_raw = load_dataset(dataset_path)
    df_feat = process_features(df_raw)

    for col in FEATURE_ORDER:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    X_num = df_feat[FEATURE_ORDER].astype("float64")
    if X_num.shape[1] != scaler.n_features_in_:
        raise RuntimeError("Nº de columnas != a lo esperado por el scaler")

    X_scaled = scaler.transform(X_num.to_numpy())

    # ---------- etiquetas 0/1 ----------
    if "label" not in df_feat.columns:
        raise RuntimeError("Falta columna 'label' en el dataset.")
    y_true = (df_feat.label.astype(str).str.upper() != "BENIGN").astype(int).to_numpy()

    # ---------- predicciones y scores ----------
    y_pred_str = model.predict(X_scaled)
    y_pred = (np.char.upper(y_pred_str.astype(str)) == "MALWARE").astype(int)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_scaled)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_scaled)[:, 1]
    else:
        scores = y_pred.astype(float)

    # ---------- Imprimir métricas globales ----------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_global = fp / (fp + tn)

    print("\n--- FlowMLDetection: general metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"ConfMat  : TN={tn}  FP={fp}  FN={fn}  TP={tp}  |  FPR={fpr_global:.4f}")

    # ---------- calcular métricas por normalización ----------
    norms = {
        "minmax": min_max_normalize,
        "robust_mm": robust_min_max,
        "ecdf": ecdf_normalize
    }
    results = []

    for norm_name, norm_func in norms.items():
        s_norm = norm_func(scores)

        auc_roc = roc_auc_score(y_true, s_norm)
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, s_norm)
        auc_pr = average_precision_score(y_true, s_norm)

        fpr_vals, tpr_vals, thr_vals = roc_curve(y_true, s_norm)
        idx_01 = np.argmin(np.abs(fpr_vals - 0.01))
        thr_01 = thr_vals[idx_01]

        yhat_01 = (s_norm >= thr_01).astype(int)
        tn1, fp1, fn1, tp1 = confusion_matrix(y_true, yhat_01, labels=[0, 1]).ravel()
        tpr_01 = tpr_vals[idx_01]
        prec_01 = precision_score(y_true, yhat_01, zero_division=0)
        rec_01 = recall_score(y_true, yhat_01, zero_division=0)
        f1_01 = f1_score(y_true, yhat_01, zero_division=0)
        acc_01 = accuracy_score(y_true, yhat_01)

        results.append({
            "normalization": norm_name,
            "strategy": "single_score",
            "threshold": thr_01,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": prec_01,
            "recall": rec_01,
            "f1": f1_01,
            "accuracy": acc_01,
            "tpr@fpr=1%": tpr_01
        })

        # ROC
        plt.figure(figsize=(6, 4))
        plt.plot(fpr_vals, tpr_vals, linewidth=2)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC: flowmldetection_{norm_name}")
        plt.grid(True, linestyle="--", alpha=0.5)
        roc_path = os.path.join(output_dir, f"flowmldetection_{norm_name}_roc.png")
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()

        # PR
        plt.figure(figsize=(6, 4))
        plt.plot(rec_vals, prec_vals, linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR: flowmldetection_{norm_name}")
        plt.grid(True, linestyle="--", alpha=0.5)
        pr_path = os.path.join(output_dir, f"flowmldetection_{norm_name}_pr.png")
        plt.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved Curves: {roc_path}, {pr_path}")

    # ---------- guardar resultados en CSV ----------
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "flowmldetection_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved Metrics in: {csv_path}")
    print(df_res)


# ----------------------------------------------------
if __name__ == "__main__":
    base_dir = "modules/slipswinae"
    module_base_dir = "modules/flowmldetection"
    dataset_path = os.path.join(base_dir, "test/friday_conn_labeled.log")
    #dataset_path = os.path.join(base_dir, "test/thursday_conn_labeled.log")
    model_path = os.path.join(module_base_dir, "model.bin")
    scaler_path = os.path.join(module_base_dir, "scaler.bin")

    for p in (dataset_path, model_path, scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    main(dataset_path, model_path, scaler_path)

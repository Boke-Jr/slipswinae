"""
Evaluación de Autoencoder (AE) + Half-Space Trees (HST) (combinación ganadora) en el espacio original de features numéricas.
Se carga el modelo AE y el HST previamente entrenados, y se evalúa la combinación AE+HST
con dos estrategias de inferencia (suma ponderada y cascada), todas fijadas a un FPR = 1 %.
Para cada esquema de normalización (Min–Max, Robust Min–Max, ECDF) se guardan métricas
en CSV y curvas ROC/PR en PNG.
"""

import os
import json
import time
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import joblib
from collections import defaultdict, deque
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# FUNCIONES DE NORMALIZACIÓN
# -----------------------------------------------------------------------------

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Min–Max normalización a [0, 1]."""
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-10)


def robust_min_max(x: np.ndarray, p_low=5.0, p_high=95.0) -> np.ndarray:
    """Robust Min–Max: recorta a percentiles p_low y p_high."""
    lower = np.percentile(x, p_low)
    upper = np.percentile(x, p_high)
    clipped = np.clip(x, lower, upper)
    return (clipped - lower) / (upper - lower + 1e-10)


def ecdf_normalize(x: np.ndarray) -> np.ndarray:
    """Normalización usando CDF empírica."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks / (len(x) - 1 + 1e-10)


# -----------------------------------------------------------------------------
# CLASE DE TESTER (AE + HST sobre features numéricas)
# -----------------------------------------------------------------------------

class ZeekFlowAEHSTTester:
    """
    Clase para evaluar la combinación AE + HST sobre las 11 características numéricas originales.
    - Carga el AE y el HST entrenados.
    - Calcula el error de reconstrucción AE para cada flujo.
    - Calcula las puntuaciones HST directamente sobre las 11 variables numéricas escaladas.
    - Para cada esquema de normalización (Min–Max, Robust Min–Max, ECDF):
      (A) Suma ponderada: grid-search sobre α ∈ {0.1,…,0.9} para maximizar F₁ @1 % FPR.
      (B) Cascada: umbral AE @1 % FPR → sobre residuos, umbral HST @1 % FPR.
      Guarda métricas en CSV y curvas ROC/PR en PNG.
    """

    def __init__(
        self,
        ae_model_path: str,
        hst_model_path: str,
        feature_info_path: str,
        labeled_file: str,
        output_dir: str = "results/ae_hst",
    ):
        """
        - ae_model_path: ruta al modelo Keras entrenado (AE).
        - hst_model_path: ruta al modelo HST entrenado.
        - feature_info_path: ruta a 'feature_info.json' con columnas ordenadas.
        - labeled_file: archivo de evaluación (JSON-lines con flujos etiquetados).
        - output_dir: carpeta donde se guardan resultados.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.hst_model_path = hst_model_path

        # 1) Cargar información de features
        with open(feature_info_path, "r") as f:
            info = json.load(f)
        self.numeric_features = info["numeric_features"]
        self.categorical_mappings = info["categorical_mappings"]
        self.expected_feature_order = info["feature_columns_ordered"]

        # 2) Cargar Autoencoder (Keras)
        self.ae_model = tf.keras.models.load_model(ae_model_path, compile=False)

        # 3) Cargar scaler para datos numéricos (el mismo usado al entrenar AE y HST)
        base_dir = os.path.dirname(ae_model_path)
        self.scaler = joblib.load(os.path.join(base_dir, "scaler.joblib"))

        # 4) Parámetros de tokenización (para procesar history)
        self.char2idx = {
            "s": 1, "h": 2, "a": 3, "d": 4, "f": 5, "r": 6, "c": 7,
            "g": 8, "t": 9, "w": 10, "i": 11, "q": 12, "^": 13, "x": 14
        }
        self.max_len = 20
        self.hist_vocab_size = 16

        # 5) Archivo de evaluación
        self.labeled_file = labeled_file

        self.history_data = None    # np.ndarray (N, max_len)
        self.numeric_data = None    # np.ndarray (N, num_features)
        self.labels = None          # np.ndarray (N,)
        self.ae_scores = None       # np.ndarray (N,)
        self.hst_scores = None      # np.ndarray (N,)

    def _encode_history(self, flows: list) -> np.ndarray:
        """
        Dada un diccionario de flujos, extrae 'history'
        y lo convierte en array entero (N, max_len).
        """
        N = len(flows)
        arr = np.zeros((N, self.max_len), dtype=np.int32)
        for i, flow in enumerate(flows):
            seq = str(flow.get("history", "")).lower()
            for j, ch in enumerate(seq[: self.max_len]):
                arr[i, j] = self.char2idx.get(ch, 0)
        return arr

    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesamiento de características idéntico al de entrenamiento.
        - Elimina columnas no usadas, combina paquetes/bytes, codifica categóricas,
          y ordena columnas según 'expected_feature_order' (para ser consistentes con el orden).
        """
        try:
            to_drop = ["uid", "smac", "dmac", "starttime", "type_", "saddr", "daddr", "history"]
            df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

            if "spkts" in df.columns and "dpkts" in df.columns:
                df["pkts"] = (
                    pd.to_numeric(df["spkts"], errors="coerce").fillna(0)
                    + pd.to_numeric(df["dpkts"], errors="coerce").fillna(0)
                )
            if "sbytes" in df.columns and "dbytes" in df.columns:
                df["allbytes"] = (
                    pd.to_numeric(df["sbytes"], errors="coerce").fillna(0)
                    + pd.to_numeric(df["dbytes"], errors="coerce").fillna(0)
                )
            df = df.drop(columns=["spkts", "dpkts", "sbytes", "dbytes"], errors="ignore")

            for feat in self.numeric_features:
                if feat not in df.columns:
                    df[feat] = 0.0
                else:
                    df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)

            for cat, mapping in self.categorical_mappings.items():
                if cat not in df.columns:
                    df[cat] = "unknown"
                df[cat] = df[cat].astype(str).str.lower().fillna("unknown")
                df[cat] = df[cat].replace("nan", "unknown")
                df[cat] = df[cat].map(mapping).fillna(0).astype(int)

            if "dir_" in df.columns:
                df["dir_"] = df["dir_"].astype(str)
                df["dir_"] = (
                    df["dir_"]
                    .replace("nan", "unknown")
                    .str.replace("<-", "in")
                    .str.replace("->", "out")
                    .str.replace("<->", "both")
                )
                df["dir_"] = df["dir_"].map(self.categorical_mappings["dir_"]).fillna(0).astype(int)

            cols_order = self.numeric_features + list(self.categorical_mappings.keys())
            for c in cols_order:
                if c not in df.columns:
                    df[c] = 0.0
            return df[cols_order].fillna(0.0)

        except Exception as e:
            print(f"Error in process_features: {e}")
            return pd.DataFrame(columns=self.numeric_features + list(self.categorical_mappings.keys()))

    def _compute_temporal(self, flows: list) -> list:
        """
        Calcula las características temporales (rolling_conn_count, dport_entropy)
        en ventana de 60 s. Igual que durante el entrenamiento.
        """
        window_time = 60
        port_histories = defaultdict(lambda: defaultdict(int))
        enriched = [None] * len(flows)
        recent_idx = deque()
        timestamps = [float(flow.get("timestamp", time.time())) for flow in flows]
        src_ips = [flow.get("saddr", "0.0.0.0") for flow in flows]
        dports = [flow.get("dport", 0) for flow in flows]
        ip_indices = defaultdict(list)

        for i, flow in enumerate(flows):
            ts = timestamps[i]
            ip = src_ips[i]
            cutoff = ts - window_time

            while recent_idx and timestamps[recent_idx[0]] < cutoff:
                old = recent_idx.popleft()
                old_ip = src_ips[old]
                if old in ip_indices[old_ip]:
                    ip_indices[old_ip].remove(old)

            recent_idx.append(i)
            ip_indices[ip].append(i)
            conn_count = len(ip_indices[ip])

            dp = dports[i]
            port_histories[ip][dp] += 1
            total = sum(port_histories[ip].values())
            entropy = 0.0
            if total > 0:
                probs = np.array(list(port_histories[ip].values())) / total
                entropy = -np.sum(probs * np.log2(probs + 1e-10))

            fc = flow.copy()
            fc["rolling_conn_count"] = conn_count
            fc["dport_entropy"] = entropy
            enriched[i] = fc

        return enriched

    def load_and_process(self):
        """
        Carga los flujos etiquetados del JSON-lines de evaluación y:
        - Normaliza formatos de entrada (Zeek→SLIPS)
        - Calcula características temporales
        - Procesa en batches los DataFrames con _process_features
        - Extrae history_data y numeric_data
        - Extrae etiquetas 0=benign, 1=malicious
        """
        if not os.path.exists(self.labeled_file):
            raise FileNotFoundError(f"The evaluation file: {self.labeled_file} does not exist.")

        with open(self.labeled_file, "r") as f:
            lines = f.read().strip().split("\n")
        flows = []
        for line in lines:
            try:
                flows.append(json.loads(line))
            except:
                continue
        if len(flows) == 0:
            raise ValueError("No flows loaded from the evaluation file.")

        field_map = {
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
        }
        for flow in flows:
            for old, new in field_map.items():
                if new not in flow and old in flow:
                    flow[new] = flow[old]
            flow.setdefault("timestamp", flow.get("ts", time.time()))
            flow.setdefault("history", "")
            flow.setdefault("label", "BENIGN")
            flow["pkts"] = int(flow.get("spkts", 0) or 0) + int(flow.get("dpkts", 0) or 0)
            flow["allbytes"] = int(flow.get("sbytes", 0) or 0) + int(flow.get("dbytes", 0) or 0)

        flows = self._compute_temporal(flows)

        batch_size = 1000
        processed = []
        for i in range(0, len(flows), batch_size):
            batch = flows[i : i + batch_size]
            df_batch = pd.DataFrame(batch)
            df_proc = self._process_features(df_batch)
            if not df_proc.empty:
                processed.append(df_proc)

        if not processed:
            raise ValueError("No valid data processed.")

        df_all = pd.concat(processed, ignore_index=True)
        actual_order = df_all.columns.tolist()
        if actual_order != self.expected_feature_order:
            df_all = df_all[self.expected_feature_order]

        self.history_data = self._encode_history(flows)         # (N, max_len)
        self.numeric_data = df_all.to_numpy(dtype=np.float32)   # (N, num_features)
        self.labels = np.array(
            [0 if str(f.get("label", "BENIGN")).upper() == "BENIGN" else 1 for f in flows],
            dtype=np.int32,
        )

        print(f"Loaded {len(flows)} samples.")
        print(f"Label distribution: {np.bincount(self.labels)} (0=benign, 1=malicious)")

    def load_hst(self):
        """
        Carga el modelo HST desde disco.
        """
        if not os.path.exists(self.hst_model_path):
            raise FileNotFoundError(f"The HST model was not found in: {self.hst_model_path}")
        self.hst_model = joblib.load(self.hst_model_path)
        print(f"[INFO] HST loaded from: {self.hst_model_path}")

    def compute_hst_scores(self):
        """
        Calcula puntuaciones de anomalía con HST sobre las 11 variables numéricas escaladas
        """
        df_num = pd.DataFrame(self.numeric_data, columns=self.expected_feature_order)
        X_num = self.scaler.transform(df_num)  # (N × 11)

        N = X_num.shape[0]
        scores = np.zeros(N, dtype=np.float64)
        feats = self.expected_feature_order
        for i, row in enumerate(X_num):
            sample_dict = {feats[j]: float(row[j]) for j in range(len(feats))}
            scores[i] = self.hst_model.score_one(sample_dict)

        self.hst_scores = scores
        print("HST scores computed over the numeric variables.")

    def compute_ae_scores(self):
        """
        Calcula el error de reconstrucción AE para cada muestra:
        - hist_rec, num_rec = ae_model.predict(...)
        - mse_num = MSE(num_rec, X_num), mae_hist = MAE(hist_rec, H_onehot)
        - ae_scores = log1p(mse_num + mae_hist)
        """
        df_num = pd.DataFrame(self.numeric_data, columns=self.expected_feature_order)
        X_num = self.scaler.transform(df_num)
        H_onehot = tf.keras.utils.to_categorical(self.history_data,
                                                 num_classes=self.hist_vocab_size)
        hist_rec, num_rec = self.ae_model.predict(
            {"history_input": self.history_data, "numeric_input": X_num},
            verbose=0
        )
        mse_num = np.mean((num_rec - X_num) ** 2, axis=1)
        mae_hist = np.mean(np.abs(hist_rec - H_onehot), axis=(1, 2))
        self.ae_scores = mse_num + mae_hist
        print("AE reconstruction errors computed")

    def _threshold_at_fpr(self, scores: np.ndarray, y_true: np.ndarray = None, target_fpr: float = 0.01) -> float:
        """
        Devuelve el umbral que aproxima un FPR = target_fpr para (y_true, scores).
        """
        if y_true is None:
            y_true = self.labels

        fpr_vals, _, thresholds = roc_curve(y_true, scores)
        if len(fpr_vals) == 0 or len(thresholds) == 0:
            uniq = np.unique(scores)
            return uniq[-1] if len(uniq) > 0 else 0.5

        idx = np.searchsorted(fpr_vals, target_fpr, side="left")
        if idx == 0:
            return thresholds[0]
        elif idx >= len(fpr_vals):
            return thresholds[-1]
        else:
            d0 = abs(fpr_vals[idx - 1] - target_fpr)
            d1 = abs(fpr_vals[idx] - target_fpr)
            return thresholds[idx - 1] if d0 <= d1 else thresholds[idx]

    def evaluate_combination(self):
        """
        Ejecuta la evaluación completa de AE + HST con dos estrategias:
        (A) Suma ponderada y (B) Cascada, en cada normalización:
        1) Cargar y procesar datos
        2) Cargar HST entrenado
        3) Calcular puntuaciones AE y HST
        4) Para cada normalización:
           (A) Grid-search α para suma ponderada, métricas, ROC/PR
           (B) Cascada: umbral AE @1 % FPR, luego HST @1 % FPR en residuales
        5) Guardar métricas en CSV y curvas ROC/PR en PNG
        """
        self.load_and_process()

        self.load_hst()

        self.compute_ae_scores()

        self.compute_hst_scores()

        norms = {
            "minmax":    min_max_normalize,
            "robust_mm": robust_min_max,
            "ecdf":      ecdf_normalize
        }
        results = []
        alphas = np.linspace(0.1, 0.9, 9)

        for norm_name, norm_func in norms.items():
            s_ae = norm_func(self.ae_scores)
            s_hst = norm_func(self.hst_scores)

            # ─── (A) Suma ponderada: grid-search α para F₁ @1 % FPR ────────────────
            best_f1 = -1.0
            best_alpha = None
            best_thr = None
            best_preds = None

            for alpha in alphas:
                fused = alpha * s_ae + (1 - alpha) * s_hst
                thr = self._threshold_at_fpr(fused, target_fpr=0.01)
                yhat = (fused >= thr).astype(int)
                f1_val = f1_score(self.labels, yhat, zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_alpha = alpha
                    best_thr = thr
                    best_preds = yhat

            # Métricas con α óptimo
            fused_opt = best_alpha * s_ae + (1 - best_alpha) * s_hst
            auc_roc = roc_auc_score(self.labels, fused_opt)
            auc_pr  = average_precision_score(self.labels, fused_opt)
            prec    = precision_score(self.labels, best_preds, zero_division=0)
            rec     = recall_score(self.labels, best_preds, zero_division=0)
            f1_val  = f1_score(self.labels, best_preds, zero_division=0)
            acc     = accuracy_score(self.labels, best_preds)

            fpr_vals, tpr_vals, _ = roc_curve(self.labels, fused_opt)
            idx = np.argmin(np.abs(fpr_vals - 0.01))
            tpr_1 = tpr_vals[idx] if len(tpr_vals) > 0 else 0.0

            results.append({
                "normalization":  norm_name,
                "strategy":       "weighted_sum",
                "alpha_ae":       best_alpha,
                "alpha_hst":      1 - best_alpha,
                "threshold":      best_thr,
                "auc_roc":        auc_roc,
                "auc_pr":         auc_pr,
                "precision":      prec,
                "recall":         rec,
                "f1":             f1_val,
                "accuracy":       acc,
                "tpr@fpr=1%":     tpr_1,
                "threshold_ae":   np.nan,
                "threshold_hst":  np.nan
            })

            # Guardar curvas ROC/PR de weighted_sum (con α óptimo)
            prefix_w = f"AE+HST_{norm_name}_weighted_α={best_alpha:.1f}"
            # ROC
            plt.figure(figsize=(6, 4))
            plt.plot(fpr_vals, tpr_vals, linewidth=2)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC: {prefix_w}")
            plt.grid(True, linestyle="--", alpha=0.5)
            roc_path = os.path.join(self.output_dir, f"{prefix_w}_roc.png")
            plt.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close()
            # PR
            p_vals, r_vals, _ = precision_recall_curve(self.labels, fused_opt)
            plt.figure(figsize=(6, 4))
            plt.plot(r_vals, p_vals, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR: {prefix_w}")
            plt.grid(True, linestyle="--", alpha=0.5)
            pr_path = os.path.join(self.output_dir, f"{prefix_w}_pr.png")
            plt.savefig(pr_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Saved Curves: {roc_path}, {pr_path}")

            # ─── (B) Cascada: umbral AE @1 % FPR → sobre residual, HST @1 % FPR ─────
            thr_ae = self._threshold_at_fpr(s_ae, target_fpr=0.01)
            mask = (s_ae >= thr_ae)
            y_pred_c = np.zeros_like(self.labels, dtype=int)
            c_scores = np.zeros_like(self.labels, dtype=float)

            if mask.any():
                hst_sub = s_hst[mask]
                labels_sub = self.labels[mask]
                if len(np.unique(labels_sub)) > 1:
                    thr_hst = self._threshold_at_fpr(hst_sub, y_true=labels_sub, target_fpr=0.01)
                else:
                    thr_hst = np.max(hst_sub)

                c_scores[mask] = hst_sub
                y_pred_c[mask] = (hst_sub >= thr_hst).astype(int)
            else:
                thr_hst = None

            auc_roc_c = roc_auc_score(self.labels, c_scores)
            auc_pr_c  = average_precision_score(self.labels, c_scores)
            prec_c    = precision_score(self.labels, y_pred_c, zero_division=0)
            rec_c     = recall_score(self.labels, y_pred_c, zero_division=0)
            f1_c      = f1_score(self.labels, y_pred_c, zero_division=0)
            acc_c     = accuracy_score(self.labels, y_pred_c)

            fpr_vals_c, tpr_vals_c, _ = roc_curve(self.labels, c_scores)
            idx_c = np.argmin(np.abs(fpr_vals_c - 0.01))
            tpr1_c = tpr_vals_c[idx_c] if len(tpr_vals_c) > 0 else 0.0

            results.append({
                "normalization":  norm_name,
                "strategy":       "cascade",
                "alpha_ae":       np.nan,
                "alpha_hst":      np.nan,
                "threshold":      np.nan,
                "auc_roc":        auc_roc_c,
                "auc_pr":         auc_pr_c,
                "precision":      prec_c,
                "recall":         rec_c,
                "f1":             f1_c,
                "accuracy":       acc_c,
                "tpr@fpr=1%":     tpr1_c,
                "threshold_ae":   thr_ae,
                "threshold_hst":  thr_hst if thr_hst is not None else np.nan
            })

            # Guardar curvas ROC/PR de cascada
            prefix_c = f"AE+HST_{norm_name}_cascade"
            # ROC
            plt.figure(figsize=(6, 4))
            plt.plot(fpr_vals_c, tpr_vals_c, linewidth=2)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC: {prefix_c}")
            plt.grid(True, linestyle="--", alpha=0.5)
            roc_path_c = os.path.join(self.output_dir, f"{prefix_c}_roc.png")
            plt.savefig(roc_path_c, dpi=150, bbox_inches="tight")
            plt.close()
            # PR
            p_vals_c, r_vals_c, _ = precision_recall_curve(self.labels, c_scores)
            plt.figure(figsize=(6, 4))
            plt.plot(r_vals_c, p_vals_c, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR: {prefix_c}")
            plt.grid(True, linestyle="--", alpha=0.5)
            pr_path_c = os.path.join(self.output_dir, f"{prefix_c}_pr.png")
            plt.savefig(pr_path_c, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Saved Curves: {roc_path_c}, {pr_path_c}")

        # Guardar resultados en CSV
        df_res = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "ae_hst_metrics.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"\n[INFO] Saved Metrics in: {csv_path}")
        print(df_res)


# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = "modules/slipswinae"
    ae_model_path     = os.path.join(base_dir, "ae.keras")
    hst_model_path    = os.path.join(base_dir, "hst.joblib")
    feature_info_path = os.path.join(base_dir, "feature_info.json")
    #labeled_file      = os.path.join(base_dir, "test/friday_conn_labeled.log")
    labeled_file      = os.path.join(base_dir, "test/thursday_conn_labeled.log")

    for path in [ae_model_path, hst_model_path, feature_info_path, labeled_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    tester = ZeekFlowAEHSTTester(
        ae_model_path=ae_model_path,
        hst_model_path=hst_model_path,
        feature_info_path=feature_info_path,
        labeled_file=labeled_file,
        #output_dir=f"{base_dir}/test/results/friday_data/ae_hst"
        output_dir=f"{base_dir}/test/results/thursday_data/ae_hst"
    )
    tester.evaluate_combination()

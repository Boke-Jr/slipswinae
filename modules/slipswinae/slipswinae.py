"""
Module for streaming anomaly detection on Zeek flow data using a combination of a Deep AutoEncoder
+ Half-Space Trees (HST). (Other tested combinations appear commented out for ilustration)
"""

from slips_files.common.imports import *
import json
import pandas as pd
from collections import deque, defaultdict
import time
import numpy as np
import joblib
import tensorflow as tf
import os
from river import anomaly
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import IsolationForest # IF model
# from sklearn.neighbors import LocalOutlierFactor # LOF model
# from sklearn.cluster import MiniBatchKMeans  # MBKMeans model
from modules.slipswinae.zeekflowae import build_zeekflow_model

# Configure mixed precision and force CPU
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.config.set_visible_devices([], 'GPU')


class SlipsWinAE(IModule):
    name = "SlipsWinAE"
    description = (
        "Anomaly detection module that uses a deep learning approach based on "
        "a Deep AutoEncoder + Half-Space Trees to detect malicious traffic."
    )
    authors = ["Antonio López Muñoz"]

    def init(self):
        """
        Método init: suscribe al canal, inicializa variables, directorio latentes.
        """
        self.c1 = self.db.subscribe("new_flow")
        self.channels = {"new_flow": self.c1}

        self.expected_features = None
        self.mode = self.read_mode()
        self.print(f"SlipsWinAE initialized in mode: {self.mode}", 0, 1)

        self.flow_window = deque(maxlen=1024)
        self.batch_size = 512
        self.processing_interval = 128
        self.flows_since_last_processing = 0
        self.batches_trained = 0
        self.minimum_flows_to_train = 50

        self.ae_model = None                    # AE (default)
        self.hst_model = None                   # HST (default)
        # self.if_model = None                  # IsolationForest
        # self.lof_model = None                 # LOF
        # self.mbk_model = None                 # MiniBatchKMeans

        self.scaler = StandardScaler()

        self.port_histories = defaultdict(lambda: defaultdict(int))
        self.window_time = 60
        self.recent_flows = deque(maxlen=10000)

        self.numeric_features = [
            'dur','sport','dport','pkts','allbytes',
            'rolling_conn_count','dport_entropy'
        ]
        self.categorical_mappings = {
            'proto':    {'tcp':1,'udp':2,'icmp':3,'Unknown':0},
            'state':    {'ESTABLISHED':1,'S0':7,'Unknown':0},
            'appproto': {'http':1,'dns':5,'Unknown':0},
            'dir_':     {'In':1,'Out':2,'Both':3,'Unknown':0}
        }
        self.training_config = {
            'ae_epochs':5, 'ae_batch_size':256, 'hst_trees':25, 'hst_height':15, 'kmeans_clusters':10
        }

        # Almacenamos latentes para posteriormente leerlos y usarlos para entrenar el IF/LOF
        #self.latents_dir = os.path.join("modules","slipswinae","latents_batches")
        #os.makedirs(self.latents_dir, exist_ok=True)
        #self._latent_batch_index = 0


    def read_mode(self):
        """
        Método read_mode: lee el modo de configuración.
        """
        conf = ConfigParser()
        return conf.get_ml_mode()


    def main(self):
        """
        Método main: recibe flujos, acumula en ventana y lanza train/inference.
        """
        msg = self.get_msg("new_flow")
        if not msg:
            return

        flow = json.loads(msg["data"]).get("flow", {})
        flow["timestamp"] = float(flow.get("ts", time.time()))
        self.flow_window.append(flow)
        self.flows_since_last_processing += 1

        if self.flows_since_last_processing >= self.processing_interval:
            if self.mode == "train":
                self._train_on_window()
            else:
                self._inference_on_window()
            self.flows_since_last_processing = 0


    def _train_on_window(self):
        """
        Método _train_on_window: procesa un batch en modo 'train', inicializa y
        entrena el AE sobre ese batch, actualiza el HST. (Otros modelos comentados por dar visibilidad)
        """
        total_start_time = time.time()

        # 1) Extraer el batch de flujos
        window_list = list(self.flow_window)
        training_flows = window_list[-self.batch_size:] if len(window_list) > self.batch_size else window_list
        n_flows = len(training_flows)

        if n_flows < self.minimum_flows_to_train and self.batches_trained == 0:
            self.print(
                f"Not enough flows ({n_flows}) to start initial training. "
                f"Need at least {self.minimum_flows_to_train}.", 0, 1
            )
            return

        self.print(f"Processing training-window of {n_flows} flows...", 0, 1)

        # 2) Construir history_data y numeric_data para el batch
        try:
            history_data, numeric_data, df_cols = self.build_arrays_for_chunk(training_flows)
            if history_data is None or numeric_data is None:
                self.print("No valid data in the current batch for processing.", 0, 1)
                return

            if self.batches_trained == 0:
                self.expected_features = df_cols
                self.print(f"Detected features for training: {self.expected_features}", 0, 1)
        except Exception as e:
            self.print(f"Error during data preparation for training: {e}", 0, 1)
            return

        # 3) Ajustar o partial‐fit del scaler
        if self.batches_trained == 0:
            self.scaler = StandardScaler()
            self.scaler.fit(numeric_data)
            self.scaler.feature_names_in_ = np.array(self.expected_features, dtype=object)
            self.print(f"Scaler initialized. Feature names stored: "
                    f"{list(self.scaler.feature_names_in_)}", 0, 1)
        else:
            if set(df_cols) != set(self.expected_features):
                self.print("Feature set changed. Re-initializing scaler.", 0, 1)
                new_scaler = StandardScaler()
                df_numeric = pd.DataFrame(numeric_data, columns=df_cols)
                for feat in self.expected_features:
                    if feat not in df_numeric.columns:
                        df_numeric[feat] = 0
                numeric_data_aligned = df_numeric[self.expected_features].to_numpy()
                new_scaler.fit(numeric_data_aligned)
                new_scaler.feature_names_in_ = np.array(self.expected_features, dtype=object)
                self.scaler = new_scaler
                numeric_data = numeric_data_aligned
            else:
                self.scaler.partial_fit(numeric_data)

        # 4) Preparar inputs para el AE
        scaled_numeric_data = self.scaler.transform(numeric_data).astype(np.float16)
        history_data_ohe = tf.keras.utils.to_categorical(history_data, num_classes=16).astype(np.float16)

        # 5) Inicializar y compilar AE si es el primer batch
        if self.ae_model is None:
            numeric_dim     = numeric_data.shape[1]
            hist_vocab_size = 16
            hist_seq_len    = history_data.shape[1]

            self.print("Initializing AE model...", 0, 1)
            self.ae_model = build_zeekflow_model(
                hist_vocab_size=hist_vocab_size,
                hist_seq_len=hist_seq_len,
                hist_enc_units=(100, 50),
                hist_latent_dim=10,
                hist_dec_units=(50, 100),
                numeric_dim=numeric_dim,
                deep_enc_layers=(128, 64, 32),
                deep_latent_dim=15,
                deep_dec_layers=(32, 64, 128),
                w_cce=1.0,
                w_mse=2.0,
                lr=3e-5
            )
            self.ae_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
                loss={
                    'history_output': 'categorical_crossentropy',
                    'numeric_output': 'mse'
                },
                loss_weights={
                    'history_output': 1.0,
                    'numeric_output': 10.0
                }
            )

        # 6) Entrenar AE
        train_history = self.ae_model.fit(
            x={'history_input': history_data, 'numeric_input': scaled_numeric_data},
            y={'history_output': history_data_ohe, 'numeric_output': scaled_numeric_data},
            batch_size=self.training_config['ae_batch_size'],
            epochs=self.training_config['ae_epochs'],
            verbose=0
        )
        last_loss = train_history.history['loss'][-1]
        self.print(f"AE training batch final loss: {last_loss:.4f}", 0, 1)

        # 7) Entrenamiento online de Half-Space Trees (HST)
        if self.hst_model is None:
            self.hst_model = anomaly.HalfSpaceTrees(
                n_trees=25,
                height=15,
                window_size=250,
                seed=42
            )
            self.print("Initialized Half-Space Trees model.", 0, 1)

        feature_names = list(self.scaler.feature_names_in_)
        for i in range(scaled_numeric_data.shape[0]):
            sample = {
                feature_names[j]: float(scaled_numeric_data[i, j])
                for j in range(scaled_numeric_data.shape[1])
            }
            self.hst_model.learn_one(sample)
        self.print(f"HST model updated with {scaled_numeric_data.shape[0]} samples.", 0, 1)

        # 8) MiniBatchKMeans partial_fit (comentado, listo para activar)
        # ----------------------------------------------------------------------------
        # if self.mb_kmeans is None:
        #     self.mb_kmeans = MiniBatchKMeans(
        #         n_clusters=self.training_config['kmeans_clusters'],
        #         random_state=42,
        #         batch_size=1024
        #     )
        # self.mb_kmeans.partial_fit(scaled_numeric_data)
        # self.print("MBKMeans partial_fit executed.", 0, 1)
        # ----------------------------------------------------------------------------

        # Guardar latentes para entrenar el IF/LOF
        #self._save_latents_to_disk(history_data, scaled_numeric_data)

        # 9) Finalizar
        self.batches_trained += 1
        total_time = time.time() - total_start_time
        self.print(f"Completed processing batch #{self.batches_trained} "
                f"in {total_time:.2f} seconds.", 0, 1)


    # def _save_latents_to_disk(self, history_data, numeric_data):
    #     """
    #     Método _save_latents_to_disk: extrae latentes AE y los guarda en .npy para entrenar el IF/LOF.
    #     """
    #     # extraer final_compressed
    #     latent_layer = next(l for l in self.ae_model.layers if l.name=="final_compressed")
    #     encoder = tf.keras.Model(inputs=self.ae_model.input,outputs=latent_layer.output)
    #     lat = encoder.predict(
    #         {'history_input':history_data,'numeric_input':numeric_data},verbose=0
    #     )
    #     fn = f"latents_batch_{self._latent_batch_index:05d}.npy"
    #     np.save(os.path.join(self.latents_dir,fn),lat)
    #     self.print(f"Saved {lat.shape[0]} latents to {fn}.",0,2)
    #     self._latent_batch_index += 1


    def _inference_on_window(self):
        """
        Método _inference_on_window: realiza inferencia (modo 'test') sobre el batch actual de flujos,
        combinando el score de reconstrucción del AE con el score de anomalía del HST usando
        normalización Robust Min–Max y umbral dinámico basado en percentil + MAD.
        """
        # 1) Verificar que el AE y el scaler estén listos
        if self.ae_model is None:
            ae_path = os.path.join("modules", "slipswinae", "ae.keras")
            if os.path.exists(ae_path):
                self.print("Loading pre-trained AE model for inference...", 0, 1)
                self.ae_model = tf.keras.models.load_model(ae_path, compile=False)
            else:
                self.print("ERROR: Pre-trained AE not found. Skipping inference.", 0, 1)
                return
        
        if self.hst_model is None:
            hst_path = os.path.join("modules", "slipswinae", "hst.joblib")
            if os.path.exists(hst_path):
                self.print("Loading pre-trained HST model for inference...", 0, 1)
                self.hst_model = joblib.load(hst_path)
            else:
                self.print("ERROR: Pre-trained HST model not found. Skipping inference.", 0, 1)
                return

        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            scaler_path = os.path.join("modules", "slipswinae", "scaler.joblib")
            if os.path.exists(scaler_path):
                self.print("Loading pre-trained scaler for inference...", 0, 1)
                self.scaler = joblib.load(scaler_path)
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.expected_features = list(self.scaler.feature_names_in_)
            else:
                self.print("ERROR: Pre-trained scaler not found. Skipping inference.", 0, 1)
                return

        # 2) Extraer el batch de flujos
        window_list = list(self.flow_window)
        inference_flows = window_list[-self.batch_size:] \
            if len(window_list) > self.batch_size else window_list
        n_flows = len(inference_flows)

        self.print(f"Performing inference on a batch of {n_flows} flows...", 0, 1)

        # 3) Preparar datos
        try:
            history_data, numeric_data, df_cols = self.build_arrays_for_chunk(inference_flows)
            if history_data is None or numeric_data is None:
                self.print("No valid data in the current batch for inference.", 0, 1)
                return

            # Alinear columnas si cambiaron respecto al scaler
            if self.expected_features is None and hasattr(self.scaler, 'feature_names_in_'):
                self.expected_features = list(self.scaler.feature_names_in_)
            if set(df_cols) != set(self.expected_features):
                df_numeric = pd.DataFrame(numeric_data, columns=df_cols)
                for feat in self.expected_features:
                    if feat not in df_numeric.columns:
                        df_numeric[feat] = 0
                numeric_data = df_numeric[self.expected_features].to_numpy()

        except Exception as e:
            self.print(f"Error preparing data for inference: {e}", 0, 1)
            return

        hist_vocab_size = 16

        # 4) Escalar y codificar
        X_num = self.scaler.transform(numeric_data)      # (N, num_features)
        H_ohe = tf.keras.utils.to_categorical(
            history_data, num_classes=hist_vocab_size
        )

        # 5) Score AE (reconstruction error)
        hist_rec, num_rec = self.ae_model.predict(
            {"history_input": history_data, "numeric_input": X_num},
            verbose=0
        )
        mse_num  = np.mean((num_rec - X_num) ** 2, axis=1)
        mae_hist = np.mean(np.abs(hist_rec - H_ohe), axis=(1,2))
        ae_score = mse_num + mae_hist

        # 6) Score HST (anomaly score: invertimos pues HST devuelve “normalidad”)
        feats = self.expected_features
        hst_raw = np.array([
            self.hst_model.score_one({ feats[j]: float(X_num[i,j]) for j in range(len(feats)) })
            for i in range(X_num.shape[0])
        ])
        hst_score = hst_raw

        # 7) Normalizar ambos con Robust Min–Max
        s_ae  = self.robust_min_max(ae_score)
        s_hst = self.robust_min_max(hst_score)

        alpha = 0.3
        fused = alpha * s_ae + (1 - alpha) * s_hst

        # 9) Umbral dinámico usando percentil + MAD
        thr = self.compute_dynamic_threshold(fused, base_percentile=95)
        y_hat = (fused >= thr)

        # 10) Alertas
        for idx, flow in enumerate(inference_flows):
            if y_hat[idx]:
                twid = flow.get("twid", "<unknown>")
                # Copiamos exactamente su forma de generar la evidencia:
                self.set_evidence_malicious_flow(flow, twid)

        tpr = y_hat.sum() / max(1, sum(1 for f in inference_flows
                                       if str(f.get("label","BENIGN")).upper()!="BENIGN"))
        self.print(f"Inference complete: flagged {y_hat.sum()} anomalies "
                   f"out of {len(inference_flows)} flows (TPR≈{tpr:.3f}, threshold={thr:.4f})",
                   0, 1)


    def set_evidence_malicious_flow(self, flow: dict, twid: str):
        """
        Emite una alerta en SLIPS igual que en FlowMLDetection:
        crea y persiste un objeto Evidence de tipo MALICIOUS_FLOW
        """
        from slips_files.common.abstracts.module import IModule
        from slips_files.core.structures.evidence import (
            Evidence, ProfileID, TimeWindow, Attacker, ThreatLevel,
            EvidenceType, IoCType, Direction, Victim, Method
        )

        confidence = 0.1
        description = (
            f"Flow with malicious characteristics by ML. "
            f"Src IP {flow['saddr']}:{flow['sport']} to {flow['daddr']}:{flow['dport']}"
        )
        # extraemos el número de ventana
        try:
            twid_number = int(str(twid).replace("timewindow", ""))
        except:
            twid_number = 0

        evidence = Evidence(
            evidence_type=EvidenceType.MALICIOUS_FLOW,
            attacker=Attacker(
                direction=Direction.SRC,
                ioc_type=IoCType.IP,
                value=flow["saddr"],
            ),
            victim=Victim(
                direction=Direction.DST,
                ioc_type=IoCType.IP,
                value=flow["daddr"],
            ),
            threat_level=ThreatLevel.LOW,
            confidence=confidence,
            description=description,
            profile=ProfileID(ip=flow["saddr"]),
            timewindow=TimeWindow(twid_number),
            uid=[ flow.get("uid", "<unknown>") ],
            timestamp=flow.get("starttime", flow.get("timestamp")) or 0,
            method=Method.AI,
            src_port=flow.get("sport"),
            dst_port=flow.get("dport"),
        )
        # finalmente, guardamos en la base de SLIPS
        self.db.set_evidence(evidence)


    def compute_dynamic_threshold(self, scores, base_percentile=95):
        """
        Método compute_dynamic_threshold: calcula un umbral dinámico basado en percentil.
        """
        return np.percentile(scores, base_percentile)


    def robust_min_max(self, x, p_low=5.0, p_high=95.0):
        lower = np.percentile(x, p_low)
        upper = np.percentile(x, p_high)
        x_clipped = np.clip(x, lower, upper)
        return (x_clipped - lower) / (upper - lower + 1e-10)


    def _compute_ae_score(self, model, history_data, numeric_data, history_data_ohe):
        """
        Método _compute_ae_score: calcula el reconstruction error del AE (MSE numérico + MAE history).
        """
        predictions = model.predict(
            {"history_input": history_data, "numeric_input": numeric_data},
            verbose=0
        )
        ae_hist_out = predictions["history_output"]
        ae_num_out = predictions["numeric_output"]

        numeric_error = np.mean((ae_num_out - numeric_data)**2, axis=1)
        history_error = np.mean(np.abs(ae_hist_out - history_data_ohe), axis=(1,2))
        return numeric_error + history_error


    def build_arrays_for_chunk(self, flows: list):
        """
        Método build_arrays_for_chunk: prepara history_data y numeric_data a partir de una lista de flujos.
        """
        numeric_rows = []
        history_rows = []
        processed_df_cols = []

        char2idx = {
            's':1,'h':2,'a':3,'d':4,'f':5,'r':6,'c':7,
            'g':8,'t':9,'w':10,'i':11,'q':12,'^':13,'x':14
        }
        max_hist_len = 20

        for flow_dict in flows:
            try:
                df_flow = pd.DataFrame([flow_dict])
                history_string = flow_dict.get("history", "")

                current_flow_ts = flow_dict.get("timestamp", time.time())
                self.recent_flows.append(flow_dict)
                self._evict_old_flows(current_flow_ts)

                src_ip = flow_dict.get("saddr", "0.0.0.0")
                rolling_conn_count = sum(1 for f in self.recent_flows if f.get("saddr") == src_ip)

                dest_port = flow_dict.get("dport", 0)
                self.port_histories[src_ip][dest_port] += 1
                total_ports_for_ip = sum(self.port_histories[src_ip].values())
                dport_entropy = 0.0
                if total_ports_for_ip > 0:
                    for count in self.port_histories[src_ip].values():
                        p = count / total_ports_for_ip
                        if p > 0:
                            dport_entropy -= p * np.log2(p)

                df_flow["rolling_conn_count"] = rolling_conn_count
                df_flow["dport_entropy"] = dport_entropy

                processed_df = self.process_features(df_flow.copy())
                if processed_df.empty:
                    self.print("Skipping flow due to processing error or empty result.", 0, 3)
                    continue

                if not processed_df_cols:
                    processed_df_cols = sorted(processed_df.columns.tolist())

                current_row_data = processed_df[processed_df_cols].iloc[0].values.astype(np.float32)
                if not np.isfinite(current_row_data).all():
                    self.print("Non-finite values in processed data, skipping flow.", 0, 2)
                    continue
                numeric_rows.append(current_row_data)

                tokens = [char2idx.get(char.lower(), 0) for char in history_string][:max_hist_len]
                tokens += [0]*(max_hist_len - len(tokens))
                history_rows.append(tokens)

            except Exception as e:
                self.print(f"Error processing a single flow: {e}. Skipping.", 0, 2)
                continue

        if not numeric_rows or not history_rows:
            return None, None, []

        numeric_data_np = np.array(numeric_rows, dtype=np.float32)
        history_data_np = np.array(history_rows, dtype=np.int32)
        return history_data_np, numeric_data_np, processed_df_cols


    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Método process_features: procesa las columnas crudas de un flujo y genera los features numéricos y categóricos.
        """
        try:
            cols_to_drop = ["uid","smac","dmac","starttime","type_","saddr","daddr","history","timestamp"]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

            if "spkts" in df.columns and "dpkts" in df.columns:
                df["pkts"] = pd.to_numeric(df["spkts"], errors='coerce').fillna(0) + \
                             pd.to_numeric(df["dpkts"], errors='coerce').fillna(0)
            if "sbytes" in df.columns and "dbytes" in df.columns:
                df["allbytes"] = pd.to_numeric(df["sbytes"], errors='coerce').fillna(0) + \
                                 pd.to_numeric(df["dbytes"], errors='coerce').fillna(0)
            df = df.drop(columns=["spkts","dpkts","sbytes","dbytes"], errors='ignore')

            for feat in self.numeric_features:
                if feat not in df.columns:
                    df[feat] = 0.0
                else:
                    df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0.0)

            for cat_feature, mapping in self.categorical_mappings.items():
                if cat_feature not in df.columns:
                    df[cat_feature] = 'Unknown'
                df[cat_feature] = df[cat_feature].astype(str).str.lower().fillna('unknown')
                df[cat_feature] = df[cat_feature].replace('nan','unknown')
                df[cat_feature] = df[cat_feature].map(mapping).fillna(mapping.get('Unknown',0)).astype(int)

            if 'dir_' in df.columns:
                df['dir_'] = df['dir_'].astype(str)
                df['dir_'] = df['dir_']\
                    .replace('nan','Unknown')\
                    .str.replace('<-','In')\
                    .str.replace('->','Out')\
                    .str.replace('<->','Both')
                df['dir_'] = df['dir_'].map(self.categorical_mappings['dir_']).fillna(0).astype(int)

            final_feature_columns = self.numeric_features + list(self.categorical_mappings.keys())
            for col in final_feature_columns:
                if col not in df.columns:
                    df[col] = 0

            return df[final_feature_columns].fillna(0)

        except Exception as e:
            self.print(f"Error during feature processing: {e}", 0, 1)
            import traceback
            self.print(traceback.format_exc(), 0, 1)
            return pd.DataFrame()


    def _evict_old_flows(self, current_ts: float):
        """
        Método _evict_old_flows: elimina de self.recent_flows los flujos cuyo timestamp < current_ts - window_time.
        """
        cutoff_ts = current_ts - self.window_time
        while self.recent_flows and self.recent_flows[0].get("timestamp", 0) < cutoff_ts:
            self.recent_flows.popleft()


    def shutdown_gracefully(self):
        """
        Método shutdown_gracefully: al apagar SLIPS en modo 'train',
        guarda AE, scaler y feature_info.
        <!-- Se mantiene el entrenamiento del IF/LOF de forma demostrativa y comentada -->
        """
        self.print("SlipsWinAE module shutting down...", 0, 1)

        if self.mode == "train":
            self.print("Saving AE, scaler and feature_info (train mode)...", 0, 1)
            model_dir = os.path.join("modules", "slipswinae")
            os.makedirs(model_dir, exist_ok=True)

            # 1) Guardar Autoencoder
            if self.ae_model is not None:
                self.ae_model.save(os.path.join(model_dir, "ae.keras"))
                self.print("Autoencoder model saved.", 0, 1)

            # 2) Guardar scaler
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
                self.print("Scaler saved.", 0, 1)

            # 3) Guardar feature_info.json
            if self.expected_features:
                feature_info = {
                    'numeric_features': self.numeric_features,
                    'categorical_mappings': self.categorical_mappings,
                    'feature_columns_ordered': self.expected_features
                }
                with open(os.path.join(model_dir, "feature_info.json"), 'w') as f:
                    json.dump(feature_info, f, indent=4)
                self.print("Feature information saved.", 0, 1)
            else:
                self.print("No feature information to save (expected_features not set).", 0, 1)

            # --------------------------------------------------------------------
            # Bulk outlier model training (commented by default)
            # Uncomment to train on all latent batches stored on disk.
            # --------------------------------------------------------------------
            # try:
            #     batch_files = sorted(os.listdir(self.latents_dir))
            #     if not batch_files:
            #         self.print("No latent batch files found. Skipping IF/LOF training.", 0, 1)
            #     else:
            #         arrays = []
            #         for fname in batch_files:
            #             if fname.endswith(".npy"):
            #                 path = os.path.join(self.latents_dir, fname)
            #                 arr = np.load(path)
            #                 arrays.append(arr)
            #         if not arrays:
            #             self.print("No valid latent arrays loaded. Aborting global training.", 0, 1)
            #         else:
            #             all_latents = np.vstack(arrays)
            #             self.print(f"Total latents for global training: {all_latents.shape}", 0, 1)
            #
            #             # a) IsolationForest global
            #             isof = IsolationForest(
            #                 n_estimators=100,
            #                 contamination="auto",
            #                 random_state=42,
            #                 n_jobs=-1
            #             )
            #             isof.fit(all_latents)
            #             joblib.dump(isof, os.path.join(model_dir, "isof_latent.joblib"))
            #             self.print("IsolationForest (global) trained and saved.", 0, 1)
            #
            #             # b) LocalOutlierFactor global
            #             lof = LocalOutlierFactor(
            #                 n_neighbors=20,
            #                 contamination="auto",
            #                 novelty=True
            #             )
            #             lof.fit(all_latents)
            #             joblib.dump(lof, os.path.join(model_dir, "lof_latent.joblib"))
            #             self.print("LocalOutlierFactor (global) trained and saved.", 0, 1)
            # except Exception as e:
            #     self.print(f"Error during IF/LOF global training: {e}", 0, 1)
            # --------------------------------------------------------------------

        self.print("SlipsWinAE module shutdown complete.", 0, 1)

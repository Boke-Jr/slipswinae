"""
Construye el modelo profundo 'ZeekFlowAE' en Keras, con:
 - AE basado en LSTM para 'history'
 - AE profundo para características numéricas
 - Suma ponderada de MSE(numérico) + CCE(history).

El modelo se usa en el módulo slipswinae.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LSTM, Embedding,
                                     TimeDistributed, RepeatVector,
                                     LeakyReLU, BatchNormalization, Concatenate)
from tensorflow.keras.models import Model

def build_zeekflow_model(
    hist_vocab_size=16,
    hist_seq_len=20,
    hist_enc_units=(100, 50),
    hist_latent_dim=10,
    hist_dec_units=(50, 100),
    numeric_dim=6,
    deep_enc_layers=(128, 64, 32),
    deep_latent_dim=15,
    deep_dec_layers=(32, 64, 128),
    w_cce=1.0,
    w_mse=10.0,
    lr=3e-5
):
    """
    Construye el modelo 'ZeekFlowAE' en Keras, con:
     - AE basado en LSTM para 'history'
     - AE profundo para características numéricas
     - Suma ponderada de MSE(numérico) + CCE(historial).

    Devuelve un modelo Keras compilado con 2 entradas y 2 salidas:
     history_output (categórico) y numeric_output (MSE).
    """

    ######### 1) AE basado en LSTM para 'history' #########
    history_input = Input(shape=(hist_seq_len,), name='history_input')
    # Embedding
    x_hist = Embedding(hist_vocab_size, 32, mask_zero=False)(history_input)

    # Codificador LSTM
    for i, units in enumerate(hist_enc_units):
        return_seq = (i < len(hist_enc_units) - 1)
        x_hist = LSTM(units, return_sequences=return_seq)(x_hist)
    # Densa para producir latente
    hist_z = Dense(hist_latent_dim, name='hist_z')(x_hist)

    ######### 2) AE profundo para características numéricas #########
    numeric_input = Input(shape=(numeric_dim,), name='numeric_input')
    x_num = numeric_input
    for layer_size in deep_enc_layers:
        x_num = Dense(layer_size)(x_num)
        x_num = LeakyReLU()(x_num)
        x_num = BatchNormalization()(x_num)
    numeric_z = Dense(deep_latent_dim, name='numeric_z')(x_num)

    ######### 3) Combinar latentes #########
    combined = Concatenate(name='combined_latent')([hist_z, numeric_z])
    # Opcionalmente, hacer una representación comprimida final
    final_compressed_dim = 25
    final_compressed = Dense(final_compressed_dim, name="final_compressed")(combined)

    ######### 4) Decodificar numérico #########
    x_dec_num = final_compressed
    for layer_size in deep_dec_layers:
        x_dec_num = Dense(layer_size)(x_dec_num)
        x_dec_num = LeakyReLU()(x_dec_num)
        x_dec_num = BatchNormalization()(x_dec_num)
    numeric_output = Dense(numeric_dim, name='numeric_output')(x_dec_num)

    ######### 5) Decodificar historial #########
    x_dec_hist = Dense(hist_dec_units[0])(final_compressed)
    x_dec_hist = RepeatVector(hist_seq_len)(x_dec_hist)

    dec_hist = x_dec_hist
    for i, units in enumerate(hist_dec_units):
        dec_hist = LSTM(units, return_sequences=True)(dec_hist)

    history_output = TimeDistributed(Dense(hist_vocab_size, activation='softmax'),
                                     name='history_output')(dec_hist)

    ######### 6) Construir y Compilar Modelo #########
    model = Model(
        inputs=[history_input, numeric_input],
        outputs=[history_output, numeric_output],
        name="ZeekFlowAE"
    )

    # Pérdidas ponderadas: w_cce * CCE(historial) + w_mse * MSE(numérico)
    losses = {
        "history_output": "categorical_crossentropy",
        "numeric_output": "mean_squared_error"
    }
    loss_weights = {
        "history_output": w_cce,
        "numeric_output": w_mse
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
          "history_output": "accuracy",  # opcional
          "numeric_output": "mse"
        }
    )

    return model

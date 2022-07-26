import numpy as np
import seaborn as sns
sns.set_style("darkgrid")

from keras.models import Model
from keras.layers import Input, LSTM, Dense, LSTMCell, RNN, Bidirectional, concatenate
import tensorflow as tf

def generate_train_sequences(x, y, input_seq_len, output_seq_len, batch_size):
    num_features = x.shape[0]
    num_labels = y.shape[0]
    # expected return shape: (batch_size, time_steps, feature_dims)
    # here a brute force way to implement but works well
    # future revision needed
    input_seq = np.zeros(shape=(batch_size, input_seq_len, num_features))
    output_seq = np.zeros(shape=(batch_size, output_seq_len, num_labels))
    for _ in range(batch_size):
        idx = np.random.randint(0, x.shape[1] - output_seq_len)
        input_seq_slice = x[:, idx:idx+input_seq_len]
        output_seq_slice = y[:, idx:idx+output_seq_len]
        for i in range(num_features):
            for j in range(input_seq_len):
                input_seq[_, j, i] = input_seq_slice[i, j]
        for i in range(num_labels):
            for k in range(output_seq_len):
                output_seq[_, k, i] = output_seq_slice[i, k]
    return input_seq, output_seq

def generate_predict_sequence(x, batch_size):
    length = x.shape[0]
    width = x.shape[1]
    res = np.zeros(shape=(batch_size, length, width))
    for i in range(batch_size):
        for j in range(length):
            for k in range(width):
                res[i][j][k] = x[j][k]
    return res

def quantile_loss(q, y, y_p):
    e = y - y_p
    return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

def create_model(n_in_features, n_out_features, en_layers, de_layers, is_bidirectional):
    # encoder
    n_en_layers = len(en_layers)
    n_de_layers = len(de_layers)
    encoder_inputs = Input(shape=(None, n_in_features))
    lstm_cells = [LSTMCell(hidden_dim) for hidden_dim in en_layers]
    if is_bidirectional:
        encoder = Bidirectional(RNN(lstm_cells, return_state=True))
        encoder_outputs_and_states = encoder(encoder_inputs)
        bi_encoder_states = encoder_outputs_and_states[1:] # [[forward_1_h, forward_1_c], [forward_2_h, ... [forward_n_h, forward_n_c], [backward_1_h, backward_1_c], [backward_2, ..., backward_n_c]]
        encoder_states = [] # we should let encoder_states = [forward_1_h, forward_1_c, forward_2_h, ... forward_n_h, forward_n_c, backward_1_h, backward_1_c, backward_2, ..., backward_n_c]
        for i in range(int(len(bi_encoder_states)/2)):
            state_h = concatenate([bi_encoder_states[i][0], bi_encoder_states[i + n_en_layers][0]], axis=-1)
            state_c = concatenate([bi_encoder_states[i][1], bi_encoder_states[i + n_en_layers][1]], axis=-1)
            temp_states = [state_h, state_c]
            encoder_states.append(temp_states)
    else:
        encoder = RNN(lstm_cells, return_state=True)
        encoder_outputs_and_states = encoder(encoder_inputs)
        encoder_states = encoder_outputs_and_states[1:]

    # decoder
    decoder_inputs = Input(shape=(None, n_out_features))
    if is_bidirectional:
        decoder_cells = [LSTMCell(hidden_dim * 2) for hidden_dim in de_layers]
    else:
        decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in de_layers]
    decoder_lstm = RNN(decoder_cells, return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_dense = Dense(n_out_features)
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


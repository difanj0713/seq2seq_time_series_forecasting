import keras
from data_wraggler import *
from dim_reduction import *
from seq2seq import *
from util import *
from arima import *
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
import keras
from numpy import genfromtxt

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    file_name = 'total_data_7064662.csv'
    train_date_axis = read_data_from_csv('train_7064662.csv')['date'].to_numpy()
    test_date_axis = read_data_from_csv('test_7064662.csv')['date'].to_numpy()

    # dimension reduction for dataset
    train_data = genfromtxt('train_x.csv', delimiter=',')
    train_target = genfromtxt('train_y.csv', delimiter=',')
    test_data = genfromtxt('test_x.csv', delimiter=',')
    test_target = genfromtxt('test_y.csv', delimiter=',')

    n_output_features = train_target.shape[0]
    in_seq_len = 30
    out_seq_len = 30
    batch_size = 128
    epochs = 30
    batches = 50

    input_seq, output_seq = generate_train_sequences(train_data, train_target, in_seq_len, out_seq_len, batch_size)
    decoder_target_data = output_seq
    decoder_input_data = np.zeros(decoder_target_data.shape)

    time_span = (test_data.shape[1] - in_seq_len)
    pred_90d = np.zeros(shape=(time_span))
    pred_360d = np.zeros(shape=(time_span))
    counter = np.zeros(shape=(time_span))
    real_seq_test = test_target[:, 0:time_span]
    actual_90d = real_seq_test.T[:, 1]
    actual_360d = real_seq_test.T[:, 0]

    acc_90d = 0
    acc_360d = 0
    d90_q_loss = 0
    d360_q_loss = 0
    quantile = 0.9
    seq2seq_model = keras.models.load_model("trained_seq2seq", compile=False)
    for i in range(time_span):
        input_seq_test = test_data[:, i:in_seq_len + i].T
        test_encoder_input_data = generate_predict_sequence(input_seq_test, batch_size)
        pred1 = seq2seq_model.predict_on_batch([test_encoder_input_data, decoder_input_data])
        pred_values1 = np.mean(pred1, axis=0)  # average through all batches
        for j in range(out_seq_len):
            if i+j < time_span:
                pred_90d[i + j] += pred_values1[:, 1][j]
                pred_360d[i + j] += pred_values1[:, 0][j]
                counter[i + j] += 1
                d90_q_loss += quantile_loss(quantile, pred_90d[i+j], actual_90d[i+j])
                d360_q_loss += quantile_loss(quantile, pred_360d[i + j], actual_360d[i + j])
        # output_seq_test1 = output_seq[0, :, 0].reshape(-1, 1)

    print("Total 90 Quantile loss for 90D:", d90_q_loss)
    print("Total 90 Quantile loss for 360D:", d360_q_loss)
    for i in range(pred_90d.shape[0]):
        pred_90d[i] = pred_90d[i] / counter[i]
        pred_360d[i] = pred_360d[i] / counter[i]

    for i in range(time_span):
        if pred_90d[i] * actual_90d[i] > 0:
            acc_90d += 1
        if pred_360d[i] * actual_360d[i] > 0:
            acc_360d += 1

    # test starting date = 2019-06-27 + 30 days = 2019-07-26
    start = dt.date(2019, 7, 26)
    then = start + dt.timedelta(days=(time_span))
    days = mdates.drange(start, then, dt.timedelta(days=1))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))
    plt.plot(days, pred_90d, label="pred_90d")
    plt.plot(days, pred_360d, label="pred_360d")
    plt.plot(days, actual_360d, label="actual_360d")
    plt.plot(days, actual_90d, label="actual_90d")
    plt.gcf().autofmt_xdate()
    plt.title("Prediction vs Actual")
    plt.ylabel("Reward", fontsize=16)
    plt.xlabel("Date", fontsize=16)
    plt.legend()
    plt.savefig('BidirectionalSeq2SeqPrediction.png')

    acc_90d /= time_span
    acc_360d /= time_span

    print("Accuracy for forward contract 90D: ", acc_90d)
    print("Accuracy for forward contract 360D: ", acc_360d)

def quantile_loss(q, y, y_p):
    e = y - y_p
    return np.max(q*e, (q-1)*e)
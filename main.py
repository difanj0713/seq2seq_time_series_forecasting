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

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.style.use('classic')

# script for testing

if __name__ == "__main__":
    target_id = 7064662
    is_first_time = False

    # don't waste time if already read and processed the data to .csv file.
    if is_first_time == True:
        # read_data(target_id)
        # train_file_name = 'train_7064662.csv'
        # test_file_name = 'test_7064662.csv'
        file_name = 'total_data_7064662.csv'
        total_data = read_data_from_csv(file_name)
        train_x, train_y, test_x, test_y = data_cleanser(total_data, file_name)
        train_x[np.isnan(train_x)] = 0
        train_y[np.isnan(train_y)] = 0
        test_x[np.isnan(test_x)] = 0
        test_y[np.isnan(test_y)] = 0
        np.savetxt("train_x.csv", train_x, delimiter=",")
        np.savetxt("train_y.csv", train_y, delimiter=",")
        np.savetxt("test_x.csv", test_x, delimiter=",")
        np.savetxt("test_y.csv", test_y, delimiter=",")

    # read file and replace nan with 0
    file_name = 'total_data_7064662.csv'
    total_data = read_data_from_csv(file_name)
    train_data, train_target, test_data, test_target = data_cleanser(total_data, file_name)
    train_data[np.isnan(train_data)] = 0
    train_target[np.isnan(train_target)] = 0
    test_data[np.isnan(test_data)] = 0 # 902*2947
    test_target[np.isnan(test_target)] = 0  # 902*2

    # dimension reduction for dataset
    train_size = train_data.shape[0]
    total_cleaned = np.concatenate((train_data, test_data), axis=0)
    total_cleaned, n_input_features = dim_reduce(total_cleaned)
    train_data = total_cleaned[:, 0:train_size] # n_input_features*train_size
    test_data = total_cleaned[:, train_size:total_cleaned.shape[1]]
    train_target = train_target.T # n_output_features*train_size
    test_target = test_target.T

    '''
    plt.clf()
    size = train_target.shape[1]
    axis = [i for i in range(size)]
    plt.plot(axis, train_target[0, :], 'r')
    plt.plot(axis, train_target[1, :], 'c')
    plt.xlabel("Date")
    plt.ylabel("Normalized Total Return")
    plt.legend(['360D', '90D'])
    plt.savefig('train_target.png')
    '''

    np.savetxt("train_x.csv", train_data, delimiter=",")
    np.savetxt("train_y.csv", train_target, delimiter=",")
    np.savetxt("test_x.csv", test_data, delimiter=",")
    np.savetxt("test_y.csv", test_target, delimiter=",")
    tft_target = np.concatenate([train_target.T, test_target.T], axis=0)
    np.savetxt("tft_target.csv", tft_target, delimiter=',')

    print("Choose your model:")
    model_selection = 'seq2seq'
    if model_selection == "seq2seq":
        # some hyperparameters
        n_output_features = train_target.shape[0]
        in_seq_len = 30
        out_seq_len = 30
        batch_size = 128
        epochs = 32
        batches = 50
        en_layers = [200, 100, 50]  # dimension of hidden layers for encoder
        de_layers = [200, 100, 50]  # dimension of hidden layers for decoder

        # seq2seq model
        quantile = 0.90
        is_bidirectional = False
        seq2seq_model = create_model(n_input_features, n_output_features, en_layers, de_layers, is_bidirectional)
        #seq2seq_model.compile(Adam(), loss='mean_squared_error')
        seq2seq_model.compile(Adam(), loss=lambda y, y_p:quantile_loss(quantile, y, y_p))
        seq2seq_model.summary()

        loss = []
        val_loss = []
        seq2seq_model.summary()
        for _ in range(batches):
            input_seq, output_seq = generate_train_sequences(train_data, train_target, in_seq_len, out_seq_len, batch_size)

            encoder_input_data = input_seq
            decoder_target_data = output_seq
            decoder_input_data = np.zeros(decoder_target_data.shape)

            history = seq2seq_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_split=0.2,
                                shuffle=False)

            loss.append(history.history['loss'])
            val_loss.append(history.history['val_loss'])

        total_loss = [j for i in loss for j in i]
        total_val_loss = [j for i in val_loss for j in i]
        # plot_loss(total_loss, total_val_loss)
        seq2seq_model.save("trained_seq2seq")
        # reconstructed_model = keras.models.load_model("trained_seq2seq")

        # post-process
        input_seq_test = test_data[:, 0:in_seq_len].T
        real_seq_test = test_target[:, 0:out_seq_len]
        test_encoder_input_data = generate_predict_sequence(input_seq_test, batch_size)
        # decoder_input_test = np.zeros((1, seq_len, n_output_features))

        pred1 = seq2seq_model.predict_on_batch([test_encoder_input_data, decoder_input_data])

        pred_values1 = np.mean(pred1, axis=0) # average through all batches
        pred_90d = pred_values1[:, 1]
        pred_360d = pred_values1[:, 0]
        # output_seq_test1 = output_seq[0, :, 0].reshape(-1, 1)
        actual_90d = real_seq_test.T[:, 1]
        actual_360d = real_seq_test.T[:, 0]

        test_df = read_data_from_csv('test_7064662.csv')
        timeline = np.array(test_df['date'][0:out_seq_len]).reshape(out_seq_len, )

        # test starting date = 2019-06-27
        start = dt.date(2019, 6, 27)
        then = start + dt.timedelta(days=out_seq_len)
        days = mdates.drange(start, then, dt.timedelta(days=1))

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
        plt.plot(days, pred_90d, label="pred_90d")
        plt.plot(days, pred_360d, label="pred_360d")
        plt.plot(days, actual_360d, label="actual_360d")
        plt.plot(days, actual_90d, label="actual_90d")
        plt.gcf().autofmt_xdate()
        plt.title("Prediction vs Actual")
        plt.ylabel("Reward", fontsize=16)
        plt.xlabel("Date", fontsize=16)
        plt.legend()
        plt.savefig('Seq2SeqPrediction.png')

    if model_selection == 'arima':
        Fwd360D_train = train_target[0, :].reshape(-1, 1)
        Fwd90D_train = train_target[1, :].reshape(-1, 1)
        naive_arima(Fwd90D_train)

        Fwd360D_test = test_target[0, :]
        Fwd90D_test = test_target[1, :]

    else:
        pass


    '''
    plt.plot(train_date_axis, train_target[0, :], 'r')
    plt.plot(train_date_axis, train_target[1, :], 'c')
    plt.xlabel("Date")
    plt.ylabel("Normalized Total Return")
    plt.legend(['360D', '90D'])
    plt.savefig('train_target.png')
    '''
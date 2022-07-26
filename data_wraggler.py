from dim_reduction import *
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import csv
import numpy as np
from sklearn import preprocessing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs

def read_data(target_id):
    # build a container for further use
    tmp_frame = pq.read_table('risklab_data/part-00000-tid-3595677196385730428-36b66bfa-d99b-45a6-94d3-0a8cfa9dc5ff-11631-1-c000.snappy.parquet')
    tmp_frame = tmp_frame.to_pandas()
    target = tmp_frame[tmp_frame['security_id'] == target_id]

    # read all .parquet files under the directory

    data_dir = Path('C:/Users/steph/PycharmProjects/gpu_test1/risklab_data')
    for parquet_file in data_dir.glob('*.parquet'):
        tmp_frame = pq.read_table(parquet_file)
        tmp_frame = tmp_frame.to_pandas()
        tmp_target = tmp_frame[tmp_frame['security_id'] == target_id]
        target = pd.concat([target, tmp_target], ignore_index=True)

    total_records = target.shape[0]
    target = target.sort_values('date')
    target.reset_index(inplace=True)
    target.drop('index', inplace=True, axis=1)
    train_target = target.head(total_records)
    train_target.to_csv('total_data_7064662.csv')

    '''
    # re-check; pass
    check_target = pd.read_csv('7064662.csv')
    check_target = check_target.loc[:, ~check_target.columns.str.contains('^Unnamed')]
    print(check_target)
    '''

    return target

# A special function used for generating file for temporal fusion transformer for convenience; Not related to seq2seq.
def generate_file_for_tft(file_name):
    data_frame = pd.read_csv(file_name, low_memory=False)
    data_frame.dropna(axis=1, how='all')
    data_frame.drop(columns=['filingDateFwd0Q', 'filingDateFwd3Q'], inplace=True)


def read_data_from_csv(file_name):
    df = pd.read_csv(file_name, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def df_normalization(data_frame, norm_method):
    if norm_method == 'min_max':
        # data_frame = (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())
        x = data_frame.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        data_frame = pd.DataFrame(x_scaled)
    elif norm_method == 'mean':
        data_frame.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    else:
        pass
    return data_frame

def find_differ_order(array):
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
    # Import data
    #df = pd.DataFrame(array, columns=['value'])[0:100]
    df = pd.read_csv('arima_y.csv', names=['value'], header=0)
    df = df.loc[0:100]
    '''
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.value[0:40]);
    axes[0, 0].set_title('Original Series')
    plot_acf(df.value, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.value.diff()[0:40]);
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.value.diff().diff()[0:40]);
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
    plt.savefig('FindingDifferencingOrder.png')
    '''
    '''
    y = df.value
    ## Adf Test
    print(ndiffs(y, test='adf'))
    # KPSS test
    print(ndiffs(y, test='kpss'))
    # PP test:
    print(ndiffs(y, test='pp'))
    '''
    plt.clf()
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df.value.diff());
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 5))
    plot_pacf(df.value.dropna(), ax=axes[1])
    plt.savefig('PartialAutocorrelation.png')

    plt.clf()
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df.value.diff());
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 1.2))
    plot_acf(df.value.diff().dropna(), ax=axes[1])
    plt.savefig('1stordercorrelation.png')
    return

def data_cleanser(data_frame, file_name):
    norm_method = 'min_max'

    # Fill up or delete empty features
    # delete columns with all values NaN together with meaningless ones
    data_frame.dropna(axis=1, how='all')
    data_frame.drop(columns=['security_id', 'date', 'filingDateFwd0Q', 'filingDateFwd3Q'], inplace=True)
    # deal with NaN according to the type of feature
    # build a dictionary given by feature list
    external_list = 'risklab_feature_list_external.csv'
    dict = pd.read_csv(external_list)
    dict.reset_index()
    category_dict = {}
    for index, row in dict.iterrows():
        #print(row['mask_name'], row['category'])
        key = row['mask_name']
        attribute = row['category']
        category_dict[key] = attribute

    # to be continued
    for column in data_frame:
        data_frame[column].replace(np.nan, 0)

    # Extract columns served as labels
    label_index = []
    temp_df = data_frame.copy()
    for column in data_frame:
        if category_dict[column] == 'Target':
            label_index.append(column)
            data_frame.drop(column, axis=1)
    label = temp_df[label_index].copy()

    # Normalization
    # label = df_normalization(label, norm_method)
    data_frame = df_normalization(data_frame, norm_method)

    # detrending / make the label stationary
    label = label.to_numpy()
    # find_differ_order(label)
    differ_order = 0

    label = pd.DataFrame(label, columns=['360D', '90D'])
    if differ_order == 0:
        pass
    elif differ_order == 1:
        label['360D'] = label['360D'].diff()
        label['90D'] = label['90D'].diff()
    else:
        label['360D'] = label['360D'].diff().diff()
        label['90D'] = label['90D'].diff().diff()

    # separate training and testing sets
    rows_count = data_frame.shape[0]
    t_index = 0.8
    training_size = int(rows_count * t_index)
    df = data_frame.to_numpy()
    l = label.to_numpy()

    train_x = df[0:training_size, :]
    train_y = l[0:training_size, :]
    test_x = df[training_size:rows_count, :]
    test_y = l[training_size:rows_count, :]

    '''
    # export to a new csv version
    if type == 'train':
        data_frame.to_csv('train_x.csv')
        label.to_csv('train_y.csv')
    if type == 'test':
        data_frame.to_csv('test_x.csv')
        label.to_csv('test_y.csv')
    '''

    return train_x, train_y,  test_x, test_y
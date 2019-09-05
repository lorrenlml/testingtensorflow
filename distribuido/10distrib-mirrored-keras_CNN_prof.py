#batch!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

#DISTRIBUTED STRATEGY IN KERAS
import tensorflow as tf


# This file creates the trained models for a given neural network configuration
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.models import model_from_json, load_model
import keras
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import sys
import os
import os.path
import json
import optparse
import time
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
#from matplotlib import style
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#chrome tracing
from tensorflow.python.client import timeline



strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


numberPunctos = 50

sensor_data = pd.DataFrame({
    "GHI AP1": [17, 31],
    "GHI AP4": [41, 22],
    "GHI AP5": [25,12],
    "GHI AP6": [41,0],
    "GHI AP7": [49,45],
    "GHI DH1": [0,49],
    "GHI DH2": [9,43],
    "GHI DH3": [13,29],
    "GHI DH4": [11,33],
    "GHI DH5": [15,37],
    "GHI DH6": [2,25],
    "GHI DH7": [1,41],
    "GHI DH8": [2,15],
    "GHI DH9": [1,31],
    "GHI DH10": [8,25],
    "GHI DH11": [10,15]
        }, index = ["latitude", "longitude"])

latlon = sensor_data.values.transpose()
gridx = np.arange(0.0, numberPunctos, 1)
gridy = np.arange(0.0, numberPunctos, 1)

def addOptions(parser):
    parser.add_option("--NNfile", default="",
                      help="Config json file for the data to pass to the model")


parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.NNfile:
    print(sys.stderr, "No configuration file specified\n")
    sys.exit(1)

# with open('config.json', 'r') as cfg_file:
with open(options.NNfile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']

train_size = cfg_data['train_size']  # [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
hor_pred = cfg_data['hor_pred']  # folder_names
days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_train_days'][0]
tg = cfg_data['time_granularity']
seed = cfg_data['seed']
station = cfg_data['station']
batch_size = cfg_data['batch_size']
num_classes = cfg_data['num_classes']
epochs = cfg_data['epochs']
img_rows = cfg_data['img_rows']
img_cols = cfg_data['img_cols']
device = cfg_data['device']
device_name = cfg_data['device_name']

print('Loading dataframes...\n')
load_start = time.time()

x_original = np.load("../x_train.npy")
print(x_original.shape)
print(len(x_original))
y_original = pd.read_csv(orig_folder + '/Y_tr_val.csv')

load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds! Splitting for train and validation...\n'.format(load_min, load_sec))

forecast_prediction = []

# Since we configured our matrices with an offset we have to adjust to "jump" to the sample we want to actually predict

for hp in hor_pred:
    if hp.endswith("min"):
        hor_pred_indices = int(int(hp.replace('min', '')) * 60 / tg)
    if hp.endswith("s"):
        hor_pred_indices = int(int(hp.replace('s', '')) / tg)
    forecast_prediction.append(hp)

    y_t = y_original  # y_train y son iquals
    y_t_index = y_t.index  # devulve una array de index
    y_t_index_valid = y_t_index[(y_t_index % day_length) < (day_length - hor_pred_indices)]  # so we don't get values for the previous or next day
    y_t_indices_lost = len(y_t_index) - len(y_t_index_valid)

    print('Indices computed. {} indices lost \n.'.format(y_t_indices_lost))
    print('Building randomized y matrix with valid indices...\n')
    y_t = np.ravel(y_original.iloc[y_t_index_valid + hor_pred_indices])

    print('Building y matrix removing invalid indices for persistence model...\n')
    y_pred_persistence = np.ravel(y_original.iloc[y_t_index_valid])  # una row de dataFram combia por numpy array

    print('Building X matrix...Same thing as before...\n')

    x_t = x_original[y_t_index_valid]  # like our randomization, just picking the same indices
    x_t = x_t.reshape(x_t.shape[0], img_rows, img_cols, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)


    #SPLIT TRAIN AND TEST SETS
    #Can split 3dimensional arrays ;) like this
    cut = int(0.9*len(x_t))
    x_train, x_test = x_t[:cut,:], x_t[cut:,:]
    y_train, y_test = y_t[:cut], y_t[cut:]
    input_shape = (img_rows, img_cols, 1)

    #When training a model with multiple GPUs, you can use the extra computing power effectively by increasing the batch size.
    #In general, use the largest batch size that fits the GPU memory, and tune the learning rate accordingly.
    #BUFFER_SIZE = 10000
    
    BATCH_SIZE_PER_REPLICA = cfg_data['batch_size']
    bs = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    #COMPILE THE MODEL
    nn_model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['MAE', 'mse'])

    output_text = 'metrics_' + 'CNN_' + str(device_name) + '_b_' + str(bs)
    nn_model.summary()

    #TENSORBOARD LOGS
    log_dir = 'logs_' + str(device_name) + '_b_' + str(bs)
    tb = tf.keras.callbacks.TensorBoard(log_dir = log_dir, batch_size=bs, profile_batch = bs)


    print('Fitting...\n' + output_text + '\n')
    fit_start = time.time()
    history=nn_model.fit(x_test,y_test, batch_size=bs, epochs=epochs, verbose=1, callbacks = [tb]    )
    fit_end = time.time()
    fit_time = fit_end - fit_start
    fit_min = int(fit_time / 60)
    fit_sec = fit_time % 60
    print('Fitting completed in {} minutes {} seconds. Saving model to .h5 file \n'.format(fit_min, fit_sec))

    #SAVE THE MODEL
    model_filename = 'CNNmodel_' + device + '_' + device_name + '_b' + str(bs) + '.h5'
    nn_model.save(model_filename)
    print("Saved model to disk")

    #INFERENCE
    print('INFERENCE')
    eval_start = time.time()
    loss_inf = nn_model.evaluate(x_test, y_test)
    eval_end = time.time()
    eval_time = eval_end - eval_start
    eval_min = int(eval_time / 60)
    eval_sec = eval_time % 60

    loss_df = pd.DataFrame(history.history['loss'])
    fit_time_df = pd.DataFrame([fit_time])
    eval_time_df = pd.DataFrame([eval_time])
    loss_inf_df = pd.DataFrame([loss_inf[0]])
    #mse_inf_df = pd.DataFrame([mse_inf])
    batch_size_df = pd.DataFrame([BATCH_SIZE_PER_REPLICA])
    epochs_df = pd.DataFrame([epochs])
    num_classes_df = pd.DataFrame([num_classes])
    img_rows_df = pd.DataFrame([img_rows])
    img_cols_df = pd.DataFrame([img_cols])
    device_df = pd.DataFrame([device])
    device_name_df = pd.DataFrame([device_name])

    epoch_time = fit_time/epochs
    epoch_time_df = pd.DataFrame([epoch_time])

    #GENERATE METRICS FILE
    df_alphascores = pd.concat([loss_df,  loss_inf_df, fit_time_df, epoch_time_df, eval_time_df, batch_size_df, epochs_df, num_classes_df, img_rows_df, img_cols_df, device_df, device_name_df], axis=1, ignore_index=True)
    df_alphascores.columns = ['mse_train', 'mse_inf', 'total_time', 'epoch_time', 'inference_time', 'batch_size', 'epochs', 'num_classes', 'img_X', 'img_Y', 'device', 'device_name']
    df_alphascores.to_csv(output_text + '.csv', header=True, index=False)


print('Model and metrics generated!\n')
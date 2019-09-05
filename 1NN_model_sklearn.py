#!/usr/bin/env python3

#This file creates the trained models for a given neural network configuration

###################################################################################
# CODIGO ORININAL DE Guillermo Yepes que genera y entrena un Perceptrón multicapa #
# mediante la librería sklean                                                     #
###################################################################################

import pandas as pd
import numpy as np
import sys
import os
import json
import optparse
import time
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



def addOptions(parser):
   parser.add_option("--NNfile", default="",
             help="Config json file for the data to pass to the model")

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.NNfile:
   print >> sys.stderr, "No configuration file specified\n"
   sys.exit(1)

#with open('config.json', 'r') as cfg_file:
with open(options.NNfile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']

train_size = cfg_data['train_size'] # [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
hor_pred = cfg_data['hor_pred'] #folder_names
alpha_values = cfg_data['alpha'] #[0.0001, 0.001, 0.01, 0,1]
feature_values = cfg_data['features'] #[['dh3'], ['dh3','dh4','dh5','dh10','ap1'], ['all']]
hls = cfg_data['hls'] #we pass it as a list or int
days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_train_days'][0]
tg = cfg_data['time_granularity']
seed = cfg_data['seed']

if isinstance(hls,list):
    hls=tuple(hls)



out_folder = orig_folder + dest_folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

model_folder = out_folder+'/models'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

csvs_folder = out_folder+'/csvs'
if not os.path.exists(csvs_folder):
    os.makedirs(csvs_folder)

graphs_folder = out_folder+'/graphs'
if not os.path.exists(graphs_folder):
    os.makedirs(graphs_folder)


print('Loading dataframes...\n')
load_start = time.time()
x_original = pd.read_csv(orig_folder+'/X_tr_val.csv')
y_original = pd.read_csv(orig_folder+'/Y_tr_val.csv')
load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds! Splitting for train and validation...\n'.format(load_min,load_sec))

split_start = time.time()
#We get the number of days and split for train and validation
lenrow_original = len(x_original.values)

print('Days: {}\n'.format(days))

arr_days = np.arange(days)
ran_seed = seed #our seed to randomize data
np.random.seed(ran_seed)
np.random.shuffle(arr_days)
len_days_validation = int(round(days * 0.176470588,0))
days_validation = arr_days[0:len_days_validation]
days_train = arr_days[len_days_validation:]

#Now we take random DAYS for train and validation:
x_train = pd.DataFrame()
y_train = pd.DataFrame()
x_val_original = pd.DataFrame()
y_val_original = pd.DataFrame()
for day in days_train:
    x_train = pd.concat([x_train,x_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
    y_train = pd.concat([y_train,y_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
for day in days_validation:
    x_val_original = pd.concat([x_val_original,x_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
    y_val_original = pd.concat([y_val_original,y_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)

lencol = len(x_train.columns) #number of columns for x
lenrow = len(x_train.values)
split_end = time.time()
split_time = split_end - split_start
split_min = int(split_time / 60)
split_sec = split_time % 60
print('Splitting completed in {} minutes {} seconds. Length for train: {}\n'.format(split_min,split_sec,len(y_train)))

forecast_prediction = []
nrmse_t_final = []
nrmse_v_final = []
skill_t_final = []
skill_v_final = []


#Since we configured our matrices with an offset we have to adjust to "jump" to the sample we want to actually predict

for hp in hor_pred:
    if hp.endswith("min"):
        hor_pred_indices = int(int(hp.replace('min','')) * 60 / tg)
    if hp.endswith("s"):
        hor_pred_indices = int(int(hp.replace('s','')) / tg)
    forecast_prediction.append(hp)

#TRAIN SIZE:

    for ts in train_size:

        n_rows = int(lenrow*ts)
        print('Taking less samples for train size = {}. y length: {} \n'.format(ts,n_rows))
        y_t = y_train.sample(n_rows,random_state=seed)
        y_t_index = y_t.index.values
        y_t_index_valid = y_t_index[(y_t_index % day_length) < (day_length - hor_pred_indices)] #so we don't get values for the previous or next day
        y_t_indices_lost = len(y_t_index) - len(y_t_index_valid)
        print('Indices computed. {} indices lost \n.'.format(y_t_indices_lost))
        print('Building randomized y matrix with valid indices...\n')
        y_t = np.ravel(y_train.iloc[y_t_index_valid + hor_pred_indices])
        print('Building y matrix removing invalid indices for persistence model...\n')
        y_pred_persistence = np.ravel(y_train.iloc[y_t_index_valid])

        y_val_index = y_val_original.index.values
        y_val_index_valid = y_val_index[(y_val_index % day_length) < (day_length - hor_pred_indices)]
        y_pred_persistence_val = np.ravel(y_val_original.iloc[y_val_index_valid])
        print('Building X matrix...Same thing as before...\n')
        x_t = x_train.iloc[y_t_index_valid] #like our randomization, just picking the same indices
        x_val = x_val_original.iloc[y_val_index_valid]
        y_val = np.ravel(y_val_original.iloc[y_val_index_valid + hor_pred_indices])
#STATIONS TO SELECT:

        for ft in feature_values:
            X_t = pd.DataFrame()
            X_val = pd.DataFrame()
            
            if ft[0] == 'all':
                X_t = x_t
                X_val = x_val
            else:
                for n in range(len(ft)):

                    for i in range(lencol):

                        if x.columns[i].startswith(ft[n]):

                            X_t = pd.concat([X,x[x.columns[i]]],axis=1,ignore_index=True)
                            X_val = pd.concat([X_val,x_val[x_val.columns[i]]],axis=1,ignore_index=True)

            scrs = []
            scrs_val = []
            rmse_train_scores = []
            rmse_validation_scores = []
            rmse_train_pers_scores = []
            rmse_validation_pers_scores = []
            skill_train_scores = []
            skill_validation_scores = []
            nrmse_train_scores = []
            nrmse_validation_scores = []

            if isinstance(hls,tuple) == False:
                 if hls > 10:
                    neurons = (hls,)
                    len_hls = '1'

            if isinstance(hls,tuple) == False:
                if hls == 1:
                     neurons = int(len(X.columns)/2 + 1)
                     hls = (neurons,)
                     len_hls = '1'

            if isinstance(hls,tuple) == False:
                if hls == 2:
                     neurons = int(len(X.columns)/2 + 1)
                     hls = (neurons,neurons)
                     len_hls = '2'

            if isinstance(hls,tuple) == False:                    
                if hls == 3:
                     neurons = int(len(X.columns)/2 + 1)
                     hls = (neurons,neurons,neurons)
                     len_hls = '3'

            else:
                len_hls = str(len(hls))
            
            hls_str = str(hls).replace('(','_').replace(', ','_').replace(')','_')
            hls_neurons_str = ''
            for i in range(len(hls)):
                hls_neurons_str = hls_neurons_str + str(hls[i])+'_'

            for av in alpha_values:



                stations = ''
                if ft[0]=="all":
                    stations = "all "
                else:
                    for sta in ft:
                        stations = stations + sta + ' '
                sts = stations.replace(' ','_')
                prcnt = round(ts*0.7,2)



                output_text = '/stations_' + sts + 'for_' + hp + '_prediction_horizon_' + str(prcnt) + '_train_size_' + len_hls + '_hidden_layers_with_' + hls_neurons_str + 'neurons'
 
                print('Creating MLPregressor\n')
                nn_model = MLPRegressor(hidden_layer_sizes=hls,alpha=av)
                print('Fitting...\n'+output_text+'\n')
                fit_start = time.time()
                nn_model.fit(X_t,y_t)
                fit_end = time.time()
                fit_time = fit_end - fit_start
                fit_min = int(fit_time / 60)
                fit_sec = fit_time % 60
                print('Fitting completed in {} minutes {} seconds. Saving model to .pkl file \n'.format(fit_min,fit_sec))
                model_filename = model_folder + output_text + '_and_alpha' + str(av) + '.pkl'
                joblib.dump(nn_model, model_filename)

                print('Predicting...\n')
                y_pred_train = nn_model.predict(X_t)
                print('Validating...\n')
                y_pred_val = nn_model.predict(X_val)
                print('Getting scores\n')
                scr = nn_model.score(X_t,y_t)
                scr_val = nn_model.score(X_val,y_val)
                scrs.append(scr)
                scrs_val.append(scr_val)

                rmse_train_pers = (np.mean((y_pred_persistence - y_t) **2)) ** 0.5 #our persistence score
                rmse_val_pers = (np.mean((y_pred_persistence_val - y_val) **2)) ** 0.5
                rmse_train_pers_scores.append(rmse_train_pers)
                rmse_validation_pers_scores.append(rmse_val_pers)


                rmse_val = (np.mean((y_pred_val - y_val) **2)) ** 0.5 
                rmse_train = (np.mean((y_pred_train - y_t) **2)) ** 0.5 

                nrmse_train = rmse_train / y_t.max() * 100
                nrmse_val = rmse_val / y_val.max() * 100

                rmse_train_scores.append(rmse_train)
                rmse_validation_scores.append(rmse_val)
                nrmse_train_scores.append(nrmse_train)
                nrmse_validation_scores.append(nrmse_val)

                nrmse_t_final.append(nrmse_train)
                nrmse_v_final.append(nrmse_val)

                skill_train = (1 - rmse_train / rmse_train_pers) * 100
                skill_val = (1 - rmse_val / rmse_val_pers) * 100
                skill_train_scores.append(skill_train)
                skill_validation_scores.append(skill_val)

                skill_t_final.append(skill_train)
                skill_v_final.append(skill_val)


            print('Saving figures and .csv file\n')

            #SAVING DATA AS .CSV
            scores = pd.DataFrame(scrs)
            scores_validation = pd.DataFrame(scrs_val)
            scores_k1_validation = pd.DataFrame(rmse_validation_scores)
            scores_k1_train = pd.DataFrame(rmse_train_scores)
            scores_kc_train = pd.DataFrame(rmse_train_pers_scores)
            scores_kc_validation = pd.DataFrame(rmse_validation_pers_scores)
            scores_nrmse_train = pd.DataFrame(nrmse_train_scores)
            scores_nrmse_validation = pd.DataFrame(nrmse_validation_scores)
            scores_k1_kc_validation = pd.DataFrame(skill_validation_scores)
            scores_k1_kc_train = pd.DataFrame(skill_train_scores)
            df_alphascores = pd.concat([scores,scores_validation,scores_k1_train,scores_k1_validation,scores_kc_train,scores_kc_validation,scores_nrmse_train,scores_nrmse_validation,scores_k1_kc_train,scores_k1_kc_validation],axis=1,ignore_index=True)

            df_alphascores.columns = ['r2_train_sklearn','r2_validation_sklearn','rmse_train','rmse_validation','rmse_persistence_train','rmse_persistence_validation','nrmse_train','nrmse_validation','skill_train','skill_validation']
            df_alphascores.to_csv(csvs_folder + output_text + '.csv',header=True,index=False)
    
    

#For use with ONE ts and ONE ft set
total_scores = pd.DataFrame({'forecast_prediction':forecast_prediction,'nrmse_train':nrmse_t_final,'nrmse_validation':nrmse_v_final,'skill_train':skill_t_final,'skill_validation':skill_v_final})
total_scores.to_csv(csvs_folder + '/scores_report_for_'+len_hls+'_hidden_layers_with_'+hls_neurons_str+'neurons.csv',header=True,index=False)

print('Figures and .csv generated!\n')
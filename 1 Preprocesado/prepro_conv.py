import numpy as np
import sys
import os
import os.path
import json
import optparse
import time
import pandas as pd

    #Randomize and split the inference set according to hor_pred
    #Generate .npy file for each hp selected

    #Coge valores aleatorios de la columna de etiquetas en función del horizonte de predicción. 
    #Coge los índices de las muestras seleccionadas y los usa para seleccionar las imágenes que 
    ##tienen asociadas.
    #Tenemos que tener pandas para la seleccion primera de las etiquetas, luego solo generamos un 
    ##.npy con ese hor_pred y con la cantidad que queramos en función del valor del split
    ####PARSEAR CON EL JSON


###################
# PARSE CONNFIG #####
##################
def addOptions(parser):
    parser.add_option("--NNfile", default="",
                      help="Config json file for the data to pass to the model")
parser = optparse.OptionParser()
addOptions(parser)
(options, args) = parser.parse_args()
if not options.NNfile:
    print(sys.stderr, "No configuration file specified\n")
    sys.exit(1)

with open(options.NNfile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)
days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_train_days'][0]
tg = cfg_data['time_granularity']
hor_pred = cfg_data['hor_pred']
forecast_prediction = []
cut_1 = cfg_data['cut']
img_rows = cfg_data['img_rows']
img_cols = cfg_data['img_cols']
orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']


##################
# DATA LOAD ######
###################
print('Loading images...\n')
load_start = time.time()
x_original = np.load("x_train.npy")
print(x_original.shape)
print(len(x_original))
print('Loading tags...\n')
y_original = pd.read_csv(orig_folder + '/Y_tr_val.csv')
load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds! Splitting for train and validation...\n'.format(load_min, load_sec))

#################
# RANDOMIZATION##
#################
# Since we configured our matrices with an offset we have to adjust to "jump" to the sample we want to actually predict

for hp in hor_pred:
    if hp.endswith("min"):
        hor_pred_indices = int(int(hp.replace('min', '')) * 60 / tg)
    if hp.endswith("s"):
        hor_pred_indices = int(int(hp.replace('s', '')) / tg)
    forecast_prediction.append(hp)
   
    y_t = y_original  # y_train y son iquals
    y_t_index = y_t.index  # devulve una array de index
    # Don't get values for the previous or next day:
    y_t_index_valid = y_t_index[(y_t_index % day_length) < (day_length - hor_pred_indices)]  
    y_t_indices_lost = len(y_t_index) - len(y_t_index_valid)
    print('Indices computed. {} indices lost \n.'.format(y_t_indices_lost))
    print('Building randomized y matrix with valid indices...\n')
    y_t = np.ravel(y_original.iloc[y_t_index_valid + hor_pred_indices])
    print('Building y matrix removing invalid indices for persistence model...\n')
    y_pred_persistence = np.ravel(y_original.iloc[y_t_index_valid])  # una row de dataFram combia por numpy array
    print('Building X matrix...Same thing as before...\n')
    # like our randomization, just picking the same indices
    x_t = x_original[y_t_index_valid]  
    x_t = x_t.reshape(x_t.shape[0], img_rows, img_cols, 1)
    
    #Split: 
    cut = int(cut_1*len(x_t))
    x_train, x_test = x_t[:cut,:], x_t[cut:,:]
    y_train, y_test = y_t[:cut], y_t[cut:]
    #print(x_train.shape, x_test.shape) 
    #print(y_train.shape, y_test.shape) #Etiquetas (valores reales que debería predecir con cada muestra)
    
    name = "set_hp_" + str(hp) + "_" + str (cut_1) + "total" + ".npy"
    name2 = "tags_hp_" + str(hp) + "_" + str (cut_1) + "total" + ".npy"

    #Para cada horizonte de predicción genero un array para inferencia
    np.save(name, x_train)
    np.save(name2, y_train)

    print('Generated {} images array \n.'.format(x_train.shape))

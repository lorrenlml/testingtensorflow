import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import timeline
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import json
import optparse
import csv
import itertools
import sys


##########################################
#Get configuration from JSON##############
##########################################
def addOptions(parser):
   parser.add_option("--NNfile", default="",
             help="Config json file for the data to pass to the model")

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.NNfile:
    print(sys.stderr, "No configuration file specified\n")
    sys.exit(1)

############################
#Parse the config JSON######
############################
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
n_nets  = cfg_data['n_nets']
BATCH_SIZE = cfg_data['batch_size']
EPOCHS = cfg_data['epochs']
LEARNING_RATE = cfg_data['learning_rate']

#############################
#Parse the .csv in JSON######
#############################
days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file) #TO JSON DIRECTLY //Include in config.json
day_length = days_info['length_day'][0]
days = days_info['number_train_days'][0]
tg = cfg_data['time_granularity']
seed = cfg_data['seed']

if isinstance(hls,list):
    hls=tuple(hls)

#####################################
#OUTPUT: generate folders if are not#
#####################################
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

###########################################
#Specify hardware to use by Tensorflow#####
###########################################
#with tf.device('/gpu:0'): 

#########################
#DATA PREPROCESSING######
#########################

#####################
#Import data#########
#####################
print('Loading dataframes...\n')
load_start = time.time()
x_original = pd.read_csv(orig_folder+'/X_tr_val.csv') 
y_original = pd.read_csv(orig_folder+'/Y_tr_val.csv')
load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds! Splitting for train and validation...\n'.format(load_min,load_sec))

##########################################
#Split into train and validation randomly#
##########################################
split_start = time.time()
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
print('The shape of input data is {} features and {} values.\n'.format(lencol,lenrow))

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
###############
#Hasta aquí tenemos la regularización de los datos

############
# TENSORES #
############
#Adaptación de las entradas a las redes 
#De pandas a numpy:
X_t, X_val = X_t.values, X_val.values
x_train = X_t 
x_test = X_val
y_train = y_t
y_test = y_val
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))
print(x_train.shape, type(x_train)) # dataframe (1366100, 50)
print(x_test.shape, type(x_test)) # dataframe (290000, 50)
print(y_train.shape, type(y_train)) # ndarray   (1366100, )
print(y_test.shape, type(y_test)) # ndarray   (290000, )


###############
# NEURAL NETS #
###############
def fully_connected_model(input_tensor, hidden_layers):
    x = tf.layers.flatten(input_tensor)
    for layer in hidden_layers:
        x = tf.layers.dense(x, layer)
        x = tf.nn.relu(x)
    # Finally, connect the output to the right shape (1 output):
    x = tf.layers.dense(x, 1)
    return x

def generador(red_1, n_redes = 1):
    '''genera n_1 redes del doble de neuronas que la red introducida'''
    n_1 = np.array(red_1)
    n_12 =np.copy(n_1)
    for x in range(0, n_redes-1):
        n_12 = n_12*np.array([2])
        n_1 = np.append(n_1, n_12)
    l = len(red_1)    
    n_1 = np.reshape(n_1, (n_redes, l))
    return n_1

###########
# Trainer #
###########   
def trainer(BATCH_SIZE, LEARNING_RATE, LOGDIR, OPT, MODEL, NN, EPOCHS, NAME, FORCE_CPU=False):
    if not os.path.isdir('log3/'):
        os.mkdir('log3/')
    if not os.path.isdir(LOGDIR):
        os.mkdir(LOGDIR)
    if not os.path.isdir(LOGDIR + '/traces/'):
        os.mkdir(LOGDIR + '/traces/')
    with open(LOGDIR + "training.log", 'w') as _out:
        tf.reset_default_graph()

        #Configure input:
        input_tensor = tf.placeholder(tf.float32, (None, x_train.shape[1]), name="input_measures")
        input_labels = tf.placeholder(tf.float32, (None, y_train.shape[1]), name="input_labels") 

        # Build the model:
        logits = MODEL(input_tensor, NN)

        #TENGO QUE SACAR LAS METRICAS DEL MODELO
        #RESHAPE DE LAS ETIQUETAS PARA COMPARAR CORRECTAMENTE LOS ARRAYS
        #¿Que sale del modelo, qué forma tiene input_labels

        # Calculate the loss function:
        #loss = tf.losses.sparse_softmax_cross_entropy(labels=input_labels, logits=logits)
        ##############################################################################################
        mse = tf.losses.mean_squared_error(labels=input_labels, predictions=logits)
        ##############################################################################################
        tf.summary.scalar("MSE",mse)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Optimizer:
        optimizer = OPT(LEARNING_RATE)
        train_step = optimizer.minimize(mse, global_step = global_step)
        summary = tf.summary.merge_all()

        # How many trainable parameters are in this model? Turns out, it's not so hard to check:
        total_parameters = 0
        for variable in tf.trainable_variables():
            this_variable_parameters = np.product([s for s in variable.shape])
            total_parameters += this_variable_parameters
            _out.write("{} has shape {} and {} total paramters to train.\n".format(
                variable.name,
                variable.shape,
                this_variable_parameters
            ))

        _out.write( "Total trainable parameters for this network: {} \n".format(total_parameters))
        
        # Here we can set up an optimizer for training the different models, and we will use profiling as well.  


        # Some of this is coming from this stack overflow answer: 
        # https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470


        # This configuration will allow us see the GPU usage with nvidia-smi.  That is, it won't hog the whole GPU. 
        if FORCE_CPU:
            #print location of each operation + pick best place for operation + forced to CPU
            config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, device_count = {'GPU': 0})
        else:
            config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        writer  = tf.summary.FileWriter(LOGDIR + '/train/') #FileWriter para albergar los ficheros para TensorBoard, iremos añadiendo
        saver   = tf.train.Saver() #Saver para guardar los ficheros paso a paso

        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph) #AÑADE GRÁFICO DE SESION

        # Allow the full trace to be stored at run time.
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)


        #LOOP DE ENTRENAMIENTO
        #Compute how many steps per epoch (based on batch size), then perform that number of steps.
        steps_per_epoch = int(len(y_train) / BATCH_SIZE)
        _out.write( "Beginning training, {} steps per epoch".format(steps_per_epoch))
        average_epoch_time = 0.0

        ############
        # Training #
        ############
        # Let's get the training start time:
        training_start_time = time.time()
        for e in range(EPOCHS):
            epoch_start_time = time.time()
            data_access_index = 0
            # We'll also get an average of time per training step:
            training_step_cummulative_time = 0.0
            n_training_steps = 0
            profiling_step_cummulative_time = 0.0
            n_profiling_steps = 0

            for step in range(steps_per_epoch):

                # Run the global step operation:
                current_global_step = session.run(global_step)

                # Construct a feed dict FROM DATA
                fd = {
                    input_tensor : x_train[data_access_index:data_access_index+BATCH_SIZE],
                    input_labels : y_train[data_access_index:data_access_index+BATCH_SIZE]
                }

                # Increment the the data_access_index:
                data_access_index += BATCH_SIZE

                # On the very first step, let's make a summary:
                if step == 0:
                    _out.write( "Generating a summary at global step {}".format(current_global_step))

                    # Create a fresh METADATA object:
                    run_metadata = tf.RunMetadata()

                    # Run the training step plus the summary:
                    start = time.time()
                    _, summ = session.run([train_step, summary], feed_dict=fd, options=run_options, run_metadata=run_metadata)
                    profiling_step_cummulative_time += time.time() - start
                    n_profiling_steps += 1

                    # Add the summary to tensorboard:
                    # (Here, we'll add things as a function of current epoch number)
                    epoch_number = e + 1.0*step / steps_per_epoch  
                    writer.add_summary(summ, epoch_number) #ADD SUMMARY TO writer
                    # And, add the run metadata with a NAME and a step:
                    writer.add_run_metadata(run_metadata, NAME + "step_{}".format(current_global_step),current_global_step) #ADD METADATA TO writer

                    # This saves the timeline to a chrome trace format:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(LOGDIR + '/traces/timeline_{}.json'.format(current_global_step), 'w') as f:
                        f.write(chrome_trace)
                else:
                    start = time.time()
                    session.run(train_step, feed_dict=fd, options=run_options, run_metadata=run_metadata)
                    training_step_cummulative_time += time.time() - start
                    n_training_steps += 1

            epoch_end_time = time.time()
            average_epoch_time += epoch_end_time - epoch_start_time

            # Compute the loss of the most recent batch:
            #current_loss = session.run(loss, feed_dict=fd)
            #_out.write( "Compeleted epoch {} of {}, current loss is {:.4f}".format(e,EPOCHS, current_loss))

            current_mse = session.run(mse, feed_dict=fd)
            _out.write( "Compeleted epoch {} of {}, current mse is {:.4f}".format(e,EPOCHS, current_mse))
            _out.write( "  Epoch training time was {:.4f}".format(epoch_end_time - epoch_start_time))
            _out.write( "  Average time per train step: {:.4f}".format(training_step_cummulative_time / n_training_steps))
            _out.write( "  Average time per prof. step: {:.4f}".format(profiling_step_cummulative_time / n_profiling_steps))
            _out.write( "-------------------------------------")

        training_end_time = time.time()

        # Save the trained model so we can reload and profile the inference later:
        saver.save(session,LOGDIR + "/train/checkpoints/save", global_step=current_global_step)


        #############
        # INFERENCE #
        #############
        ##LOOP DE inferencia
        #Compute how many steps per epoch (based on batch size), then perform that number of steps.
        steps_per_epoch_i = int(len(y_test) / BATCH_SIZE)
        inference_start_time = time.time()
        #tf.Variable('predictions', predictions)
        #predictions = np.empty(shape = (len(y_test),1)) 

        #Creo un array enganche al que le voy incluyendo los valores del tensor de inferencia
        predictions = np.array([1])
        predictions = predictions.reshape(1,1)
        ytest = np.append(predictions, y_test)
        for step in range(steps_per_epoch_i):
            data_access_index_i = 0
            # Run the global step operation:
            #current_global_step = session.run(global_step)
            # Construct a feed dict FROM DATA
            fd_i = {
                input_tensor : x_test[data_access_index_i:data_access_index_i+BATCH_SIZE],
                input_labels : y_test[data_access_index_i:data_access_index_i+BATCH_SIZE]
            }
            pred_1000 = session.run(logits, feed_dict = fd_i, options=run_options, run_metadata=run_metadata)
            predictions = np.append(predictions, pred_1000, axis=0)
            # Increment the the data_access_index:
            data_access_index_i += BATCH_SIZE 
        #Elimino el enganche y tengo el tensor de inferencia
        predictions = predictions[1:,]
        # While the graph is open and loaded, let's run inference on the test set to evaluate performance
        run_metadata = tf.RunMetadata()        
                             
        writer.add_run_metadata(run_metadata, NAME + "inference",current_global_step) #ADD METADATA TO writer
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        #Add chrome trace:
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(LOGDIR + '/traces/timeline_inference.json', 'w') as f:
            f.write(chrome_trace)
        inference_end_time = time.time()
        session.close()

        # Return some metrics:
        return predictions


################################
# GENERATE LOGS & METRICS#######
################################

#############################################################################
global_results = dict()
n_1 = generador(hls, n_nets)
print('Nets are these:' +  str(n_1))   ##
##############################################################################

#Aquí generamos las carpetas con los logs y las traces para cada uno de los modelos en CPU o GPU llamando a trainer.
#Las identificamos convenientemente.
#for network in ['fully_connected', 'light_connected', 'convolutional']:
for network in ['fully_connected']:
    for opt in ['ADAM']:#, 'SGD']:
        for n  in n_1: #LLAMADA A FUNCION DE DEVUELVE EL ARRAY DE LAS REDES
            for device in ['CPU', 'GPU']:
                start = time.time()
                NAME = network + "_" + str(n) + "_" + device
                if opt == 'ADAM':
                    OPT = tf.train.AdamOptimizer
                else:
                    OPT = tf.train.GradientDescentOptimizer
                
                LOGDIR = 'log3/' + NAME
                if network == 'fully_connected':
                    MODEL = fully_connected_model
                    NN = n
                elif network == 'light_connected':
                    MODEL = light_connected_model
                else:
                    MODEL = convolutional_neural_network
                    
                if device == 'CPU':
                    results = trainer(BATCH_SIZE, LEARNING_RATE, LOGDIR, OPT, MODEL, NN, EPOCHS, NAME, FORCE_CPU=True)
                else:
                    results = trainer(BATCH_SIZE, LEARNING_RATE, LOGDIR, OPT, MODEL, NN, EPOCHS, NAME)
                            
print(results[:100,], y_test[:100,])

metrics = pd.DataFrame({'predictions':[results[:100,]], 'labels': [y_test[:100,]]})
metrics.to_csv('inferencia.csv')

print(results.shape) #Faltan esas 790 muestras del último 'step' que no se hace en la inferencia
print(y_test.shape) 


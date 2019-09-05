
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import timeline
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


############
# TENSORES #
############
#Adaptación de las entradas a las redes 
#De pandas a numpy:
#X_t, X_val = X_t.values, X_val.values
#x_train = X_t 

X_val = np.load('x_input.npy')
y_val = np.load('y_input.npy')
x_test = X_val
#y_train = y_t
y_test = y_val
#y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))
#print(x_train.shape, type(x_train)) # dataframe (1366100, 50)
print(x_test.shape, type(x_test)) # dataframe (290000, 50)
#print(y_train.shape, type(y_train)) # ndarray   (1366100, )
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
# INFERENCER #
###########   
# Ahora 
#Removed OPT
def trainer(BATCH_SIZE, LEARNING_RATE, LOGDIR, MODEL, NN, EPOCHS, NAME, FORCE_CPU=False):
    if not os.path.isdir('log3/'):
        os.mkdir('log3/')
    if not os.path.isdir(LOGDIR):
        os.mkdir(LOGDIR)
    if not os.path.isdir(LOGDIR + '/traces/'):
        os.mkdir(LOGDIR + '/traces/')
    with open(LOGDIR + "training.log", 'w') as _out:
        tf.reset_default_graph()
        #Configure input: TEST VALUES
        input_tensor = tf.placeholder(tf.float32, (None, x_test.shape[1]), name="input_INFERENCE")
        input_labels = tf.placeholder(tf.float32, (None, y_test.shape[1]), name="input_INFERENCE_labels") 
        # Build the model:
        logits = MODEL(input_tensor, NN)
        mse = tf.losses.mean_squared_error(labels=input_labels, predictions=logits)
        tf.summary.scalar("MSE",mse)
        # Global Step variable
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        summary = tf.summary.merge_all()
        # This configuration will allow us see the GPU usage with nvidia-smi.  That is, it won't hog the whole GPU. 
        if FORCE_CPU:
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

        #############
        # INFERENCE #
        #############
        ##LOOP DE inferencia
        #Compute how many steps per epoch (based on batch size), then perform that number of steps.
        steps_per_epoch_i = int(len(y_test) / BATCH_SIZE)
        inference_start_time = time.time()
        data_access_index_i = 0
        for step in range(steps_per_epoch_i):
            # Run the global step operation:
            current_global_step = session.run(global_step)
            # Construct a feed dict FROM DATA
            fd_i = {
                input_tensor : x_test[data_access_index_i:data_access_index_i+BATCH_SIZE],
                input_labels : y_test[data_access_index_i:data_access_index_i+BATCH_SIZE]
            }
            #ACTUALIZA CON VALOR DE OPERACIÓN TRAS CADA 'step'
            # Increment the the data_access_index:
            data_access_index_i += BATCH_SIZE

        # While the graph is open and loaded, let's run inference on the test set to evaluate performance
        run_metadata = tf.RunMetadata()   
        
           
        #NO HACEMOS RUN SOBRE 'train_step'(optimizador) a diferencia del entrenamiento
        mse= session.run(mse, feed_dict = fd_i, options=run_options, run_metadata=run_metadata)
    
    
        writer.add_run_metadata(run_metadata, NAME + "inference",current_global_step) #ADD METADATA TO writer
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        #Add chrome trace:
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        
        with open(LOGDIR + '/traces/timeline_inference.json', 'w') as f:
            f.write(chrome_trace)
        inference_end_time = time.time()
        # Save the trained model so we can reload and profile the inference later:
        #saver.save(session,LOGDIR + "/train/checkpoints/save", global_step=current_global_step)
        session.close()

        # Return some metrics:
        return {
            'mse'       : mse,
            'inference_time' : inference_end_time - inference_start_time
        }


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
                # if opt == 'ADAM':
                #     OPT = tf.train.AdamOptimizer
                # else:
                #     OPT = tf.train.GradientDescentOptimizer
                
                LOGDIR = 'log3/' + NAME
                if network == 'fully_connected':
                    MODEL = fully_connected_model
                    NN = n
                elif network == 'light_connected':
                    MODEL = light_connected_model
                else:
                    MODEL = convolutional_neural_network
                    
                if device == 'CPU':
                    results = trainer(BATCH_SIZE, LEARNING_RATE, LOGDIR, MODEL, NN, EPOCHS, NAME, FORCE_CPU=True)
                else:
                    results = trainer(BATCH_SIZE, LEARNING_RATE, LOGDIR, MODEL, NN, EPOCHS, NAME)
                            
                results['name'] = NAME
                results['batch_size'] = BATCH_SIZE
                results['learning_rate'] = LEARNING_RATE
                results['opt'] = opt
                results['model'] = network
                results['epochs'] = EPOCHS
                global_results[NAME] = results
                print ("Finished training {} in {:.2f}s".format(NAME, time.time() - start))


#SACAMOS LOS DATOS DE AQUÍ
#pARSEADO DEL DICCIONARIO A TABLA CON EL NOMBRE COMO PRIMERA COLUMNA
fields = ['net',  'mse', 'inference_time', 'parameters', 'name', 'batch_size', 'learning_rate', 'opt', 'model', 'epochs']
with open('metrics.csv', 'w', newline='') as csvfile:
    w = csv.DictWriter(csvfile, fields)
    for key,val in sorted(global_results.items()):
        row = {'net': key}
        row.update(val)
        w.writerow(row)
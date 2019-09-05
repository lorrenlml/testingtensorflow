import pandas as pd 
import numpy as np

#############################
#Indicate % of data         #
CUT = 80  
#############################

#Import pandas and convert to numpy
train = pd.read_csv('X_tr_val.csv').values
train2 = pd.read_csv('Y_tr_val.csv').values

#SPLIT with same length the data in two datasets
cut = int((CUT/100)*len(train)) #80% data
train = train[:cut,:]
train2 = train2[:cut,:]
print (train.shape)
print (train2.shape)
print('Flatten etiquetas to get shape (X,) no (X,1)')

#Flatten tags column to get neccesary shape for model input
train2 = train2.flatten()
print (train.shape)
print (train2.shape)

np.save('all_samples.npy', train)
np.save('all_tags.npy', train2)
print('.npy lifes generated with desired length')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:12:54 2023

@author: smfen
"""

# DSCI 441 Project

import pandas as pd
#from dataprep.eda import create_report
#from pandas_profiling import ProfileReport
import numpy as np
#from os import listdir 
import wfdb #This package will likely need to be pip installed.
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
#from keras.utils import to_categorical
from keras.layers import Conv1D
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR

#specify data path to directory with full annotations
path = 'mit-bih-arrhythmia-database-1.0.0/'

#This time, split it up by patients. I went through the zip file and\
# and examined the patients number directly and marked them down. 
patients = ['100', '101', '102', '103', '104', '105', '106', '107', 
           '108', '109', '111', '112', '113', '114', '115', '116', 
           '117', '118', '119', '121', '122', '123', '124', '200',
           '201', '202', '203', '205', '207', '208', '209', '210', 
           '212', '213', '214', '215', '217', '219', '220', '221', 
           '222', '223', '228', '230', '231', '232', '233', '234']

data = pd.DataFrame()

# Here we are extracting the individual files from the zip file based on patient
for pat in patients:
    file = path + pat
    annotation = wfdb.rdann(file, 'atr')
    sym = annotation.symbol
    
    values, counts = np.unique(sym, return_counts=True)
    data_sub = pd.DataFrame({'sym' :values, 'val':counts, 'pat':[pat]*len(counts)})
    data = pd.concat([data, data_sub], axis=0)
# The reason we moved to these data was for the larger classification
# Now we see the 23 groups, instead of 5 from the previous data above.
print(data.groupby('sym').val.sum().sort_values(ascending = False))

#Some of these don't represent beats though, so let's sort
# Here is the cheat sheet: https://archive.physionet.org/physiobank/annotations.shtml
nbeat = ['Q', '?', '[', ']', '!', 'x', '(', ')', 'p', 't', 'u', '`', 
         '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
arry = ['L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 
       'n', 'E', 'f', '/']

# A labeled beat of 'N' is sinus rythm = normal 
data['cat'] = -1 #Not a beat
data.loc[data.sym=='N', 'cat'] = 0 # Sinus rythm 
data.loc[data.sym.isin(arry), 'cat'] =1 # Arrhythmia beat
# Print out number of each classification
print(data.groupby('cat').val.sum())

#Now we must load that actual ECG file. Classifications were above.
def load_ecg(file):
    #load the actual ecg file
    record = wfdb.rdrecord(file)
    # annotation
    anno = wfdb.rdann(file, 'atr')
    # get signal
    p_sig = record.p_signal
    # make sure the frequency is 360, if not could run into issue
    assert record.fs == 360, 'sample freq is not 360'

    # Symbols and annotation index
    atr_sym = anno.symbol
    atr_samp = anno.sample
    
    return p_sig, atr_sym, atr_samp

#Examining an example patient and the type of annotated beats 
file = path + patients[0]
p_sig, atr_sym, atr_samp = load_ecg(file)
values, counts = np.unique(sym, return_counts=True)
for v,c in zip(values, counts):
    print(v,c)
    
#Let's view some of the abnormal beats
ab_index = [b for a,b in zip(atr_sym, atr_samp) if a in arry][:20]
print(ab_index)

x = np.arange(len(p_sig))

left = ab_index[5]-1080
right = ab_index[5]+1080

plt.plot(x[left:right], p_sig[left:right,0], '-', label='ecg',)
plt.plot(x[atr_samp], p_sig[atr_samp,0], 'go', label='normal')
plt.plot(x[ab_index], p_sig[ab_index,0], 'ro',label='abnormal')

plt.xlim(left,right)
plt.ylim(p_sig[left:right].min()-0.05, p_sig[left:right,0].max()+0.05)
plt.xlabel('time index')
plt.ylabel('ECG signal')
plt.legend(bbox_to_anchor = (1.04, 1), loc = 'upper left')
plt.show()

# Now we actually have to split the dataset into individual beats. 
def make_dataset(patients, num_sec, fs, arry):
    #Need to make the dataset, but ignore non-beats (not useful to us)
    #Inputs are patients, num of sec we want before/after beat
    # frequency (fs)
    # abnoraml is a marker for what we are looking for
    # output is X_all, Y_all, sym_all
    #initialize arrays
    num_cols = 2*num_sec*fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []
    
    #list to keep track of number of beats across patients 
    # even though the recording lengths are similar, HR is varying so 
    # we will have different number of beats for each patient
    max_rows = []
    
    for pt in patients:
        file = path + pt
        p_sig, atr_sym, atr_samp = load_ecg(file)
        
        #grab the signal
        p_sig = p_sig[:,0]
        
        #make df to exclude the non-beats
        df_ann = pd.DataFrame({'atr_sym':atr_sym, 
                              'atr_samp':atr_samp})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(arry + ['N'])]
        
        X,Y,sym = build_XY(p_sig, df_ann, num_cols, arry)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all, X, axis=0)
        Y_all = np.append(Y_all, Y, axis =0)
    #drop the first zero row
    X_all = X_all[1:, :]
    Y_all = Y_all[1:, :]
    
    #check sizes
    assert np.sum(max_rows) == X_all.shape[0], 'number of X, max_rows rows incorrect'
    assert Y_all.shape[0] == X_all.shape[0], 'number of X, Y, rows incorrect.'
    assert Y_all.shape[0] == len(sym_all), 'number of Y, sym rows incorrect'
    return X_all, Y_all, sym_all

def build_XY(p_sig, df_ann, num_cols, arry):
    #Build X,Y matrices for each beat
    # Return symbols for Y (annoations)
    num_rows = len(df_ann)
    X= np.zeros((num_rows, num_cols))
    Y= np.zeros((num_rows, 1))
    sym=[]
    #Keep track rows
    max_row = 0
    
    for atr_samp, atr_sym in zip(df_ann.atr_samp.values, df_ann.atr_sym.values):
        left = max([0,(atr_samp-num_sec*fs)])
        right = min([len(p_sig),(atr_samp + num_sec*fs)])
        x = p_sig[left:right]
        if len(x) == num_cols:
            X[max_row,:] = x
            Y[max_row,:] = int(atr_sym in arry)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row,:]
    Y = Y[:max_row,:]
    return X, Y, sym

#This will process all of the patients with 3 sec before and after beats
num_sec = 3
fs = 360
X_all, Y_all, sym_all = make_dataset(patients, num_sec, fs, arry)

import random
random.seed(42)
pts_train = random.sample(patients, 36)
pts_valid = [pt for pt in patients if pt not in pts_train]
print(len(pts_train), len(pts_valid))

X_train, y_train, sym_train = make_dataset(pts_train, num_sec, fs, arry)
X_valid, y_valid, sym_valid = make_dataset(pts_valid, num_sec, fs, arry)
print(X_train.shape, y_train.shape, len(sym_train))
print(X_valid.shape, y_valid.shape, len(sym_valid))

## Some metrics functions
def calc_prevalence(y_act):
    return (sum(y_act)/len(y_act))

def calc_specificity(y_act, y_pred, thresh):
    return sum((y_pred < thresh) & (y_act == 0))/sum(y_act==0)

def print_report(y_act, y_pred, thresh):
    auc = roc_auc_score(y_act, y_pred)
    accuracy = accuracy_score(y_act, (y_pred > thresh))
    recall = recall_score(y_act, (y_pred > thresh))
    precision = precision_score(y_act, (y_pred > thresh))
    specificity = calc_specificity(y_act, y_pred, thresh)
    print('AUC: ', auc)
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('Specificity: ', specificity)
    print('Prevelance: ', calc_prevalence(y_act))
    print(' ')
    return auc, accuracy, recall, precision, specificity

# Now we need to start building the models. 
# Let's reshape our input for the CNN
#############CNN##################
X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid_cnn = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

print(X_train_cnn.shape)
print(X_valid_cnn.shape)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation = 'relu', input_shape = (2160,1)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

#compile
model.compile(
    loss = 'binary_crossentropy', 
    optimizer = 'adam', 
    metrics = ['accuracy'])

model.fit(X_train_cnn, y_train, batch_size = 1, epochs=1, verbose=1)

y_train_preds_cnn = model.predict(X_train_cnn, verbose=1)
y_valid_preds_cnn =  model.predict(X_valid_cnn, verbose=1)

thresh = (sum(y_train)/len(y_train))[0]

print('Train - CNN');
print_report(y_train, y_train_preds_cnn, thresh)
print('Test -- CNN');
print_report(y_valid, y_valid_preds_cnn, thresh);

##########     LSTM     ##########
from keras.layers import Bidirectional, LSTM
model_LSTM = Sequential()
model_LSTM.add(Bidirectional(LSTM(64, input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]))))
model_LSTM.add(Dropout(rate=0.20))
model_LSTM.add(Dense(1, activation = 'sigmoid'))
model_LSTM.compile(
    loss = 'binary_crossentropy', 
    optimizer = 'adam', 
    metrics = ['accuracy'])

model_LSTM.fit(X_train_cnn, y_train, batch_size=32, epochs=2, verbose=1)

y_train_preds_lstm = model_LSTM.predict(X_train_cnn, verbose=1)
y_valid_preds_lstm = model_LSTM.predict(X_valid_cnn, verbose=1)

print('Train - LSTM');
print_report(y_train, y_train_preds_lstm, thresh)
print('Test -- LSTM'); 
print_report(y_valid, y_valid_preds_lstm, thresh);

test_list = [X_valid_cnn[1]]
model_LSTM.predict(test_list)
y_train_int = [int(x) for x in y_train]
val, con = np.unique(y_train_int, return_counts=True)
for v, c in zip(val,con):
    print(v,c)
   
#### Working with original data set. I believe the above is not functioning correctly. 
# First upload the data set into Python
train = pd.read_csv("mitbih_train.csv", header=None)

# See preliminary dataframe size
print(data.size)

# Set the target variable (this is the last column of numbers in the DF)
train_y = train.iloc[:, -1]
train_y = train_y.astype('int')
train_y.head()
# Set the rest of the columns as the x-variables (all other)
train_x = train.iloc[:, :-1]
train_x.head()
# Set to floats
train_x = train_x.astype('float')
train_x.head()
# Finding the occurences of each type of beat classification. 
unique, counts = np.unique(train_y, return_counts=True)
print(f'unique values: {unique}')
print(f'counts: {counts}')
# Create dict so that each unique classification goes with class name.
class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
#plot the first 5 ecg signals in the dataset and their corresponding labels 
fig, ax = plt.subplots(5, 1, figsize=(20, 10))
for i in range(5):
    ax[i].plot(train_x.iloc[i, :])
    ax[i].set_title(f'Label: {class_names[train_y[i]]}')
    ax[i].set_xlabel('Time (ms)')
    ax[i].set_ylabel('Amplitude (mV)')
    
X_np = train_x.to_numpy()
X_np.shape

# Add a channel dimension
X_np = X_np.reshape(-1, 1, 187)
X_np.shape
y_np = train_y.to_numpy()
y_np = y_np.reshape(-1, 1)
y_np.shape

#We need to normalize the data. 
x_mean = X_np.mean()
x_std = X_np.std()
x_max = X_np.max()
X_min = X_np.min()
print('Mean before normalization: ', x_mean)
print('STD: ', x_std)
print('max :', x_max)

x_norm = X_np - x_mean
x_norm *= 1/x_std

print(x_norm.shape)
print('New mean is: ', x_norm.mean())
print('New std is: ', x_norm.std())

# Now let's visualize the new normalized data
fig, ax = plt.subplots(5, 1, figsize=(20, 10))
for i in range(5):
    ax[i].plot(x_norm[i,0,:])
    ax[i].set_title(f'Label: {class_names[train_y[i]]};')
    ax[i].set_xlabel('Time (ms)')
    ax[i].set_ylabel('mV')
    
fig, ax = plt.subplots(1, 1, figsize=(20,10))
ax.hist(x_norm.flatten(), bins=100)
ax.set_title(f'Distribution of data')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('mV')

np.savetxt('x_norm.csv', x_norm.squeeze(), delimiter=',')
np.savetxt('y_np.csv', y_np, delimiter=',')

x_norm = np.loadtxt('x_norm.csv', delimiter=',').reshape(-1,1,187)
y_np = np.loadtxt('y_np.csv', delimiter=',').reshape(-1,1)
print(x_norm.shape)
print(y_np.shape)
y_np = y_np.astype('int')
y_np.dtype

#Split into train and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_norm, y_np, test_size=0.25, shuffle=True)

#There is class imbalances here. We may need to balance the data. 
print(f'Frequency of each class in the train set')
unique, counts = np.unique(y_train, return_counts=True)
for i in range(len(unique)):
    print(f'{class_names[unique[i]]}: {counts[i]}')
print(f'Frequency of each class in the test set')
unique, counts = np.unique(y_test, return_counts=True)
for i in range(len(unique)):
    print(f'{class_names[unique[i]]}: {counts[i]}')
    
h_params = {'bs': 1024, 
            'lr': 1e-3,
            'lr_decay': 0.3,
            'epochs': 50}

train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

len(train_ds), len(test_ds)
# Here I compute the weights od the class by inversion and normalization
class_dist = np.unique(y_train, return_counts=True)[1]
class_dist = class_dist/np.sum(class_dist)
weights = 1/class_dist
weights = weights/weights.sum()
weights, class_names
# Assign weights from above the training set
data_weights = np.zeros(y_train.shape[0])
for i in range(y_train.shape[0]):
    data_weights[i] = weights[y_train[i,0]]
data_weights[:10]
#
train_dl = DataLoader(train_ds, batch_size=h_params['bs'], 
                      sampler=WeightedRandomSampler(weights=data_weights, num_samples=len(y_train), 
                                                    replacement=True))
test_dl = DataLoader(test_ds, batch_size=h_params['bs'])
#class distribs in dataloader, want to make sure they are uniform
for batch in train_dl:
    print(batch[1].shape)
    unique, counts = np.unique(batch[1], return_counts=True)
    for i in range(len(unique)):
        print(f'{class_names[unique[i]]}: {counts[i]}')
    break
# Lets examine a few random samples from the batch now
x, y = next(iter(train_dl))
fig, ax = plt.subplots(5,1,figsize=(20,10))
for i in range(5):
    ax[i].plot(x[i,0,:])
    ax[i].set_title(f'Label: {class_names[y[i].item()]}')
    ax[i].set_xlabel('Time (ms)')
    ax[i].set_ylabel('mV')
    
        
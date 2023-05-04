#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:11:33 2023

@author: smfen
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.title("Arrhythmia Classification from Electrocardiograms using Long-Short Term Memory Networks")
st.write("This app will allow you to choose a testing input (single beat .csv file) and it will output an arrythmia classification.")
    
uploaded_file = st.file_uploader("Choose input file for classification.")


model = keras.models.load_model('/Users/smfen/Documents/Lehigh Graduate School/Lehigh Spring 2023/DSCI 441 -- Stat. and ML/Project/streamlit_model')


if uploaded_file != 'NoneType':
    test_df = pd.read_csv(uploaded_file, header=None)
    features_df = test_df.iloc[:,:-1]
    features_df = features_df.astype('float32')
    X_test_np = features_df.to_numpy()
    X_test_np = np.reshape(X_test_np, (X_test_np.shape[0], 1, X_test_np.shape[1]))
    prediction = model.predict(X_test_np, verbose=1)
    prediction = (np.rint(prediction)).astype('int')
    zero = prediction[0][0]
    one = prediction[0][1]
    two = prediction[0][2]
    three = prediction[0][3]
    four = prediction[0][4]
    if zero == 1: pred = 'N - Sinus Rhythm'
    if one == 1: pred = 'S - Superventricular Premature'
    if two == 1: pred = 'V - Ventricular Premature'
    if three == 1: pred = 'F - Ventricular Fusion'
    if four == 1: pred = 'Q - Unclassifiable'

    st.write('The predicted arrhythmia is: ', pred)

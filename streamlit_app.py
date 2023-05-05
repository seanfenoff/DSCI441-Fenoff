#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:11:33 2023

@author: smfen
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

st.title("Arrhythmia Classification from Electrocardiograms using Long-Short Term Memory Networks")
st.header("This app will allow you to choose a testing input (single beat CSV file) and it will output an arrythmia classification.")
    
uploaded_file = st.file_uploader("Choose input file for classification.")


if uploaded_file is not None:
    
    test_df = pd.read_csv(uploaded_file, header=None)
    features_df = test_df.iloc[:,:-1]
    features_df = features_df.astype('float32')
    fig, ax = plt.subplots()
    ax.plot(test_df.iloc[0,:])
    ax.set_title('Plot of the input ECG Signal')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    st.pyplot(fig)
    
    @st.cache_resource
    def model_prediction(features_df):
        X_test_np = features_df.to_numpy()
        X_test_np = np.reshape(X_test_np, (X_test_np.shape[0], 1, X_test_np.shape[1]))
        model = keras.models.load_model('/Users/smfen/Documents/Lehigh Graduate School/Lehigh Spring 2023/DSCI 441 -- Stat. and ML/Project/streamlit_model')
        prediction = model.predict(X_test_np, verbose=1)
        prediction = (np.rint(prediction)).astype('int')
        zero = prediction[0][0] 
        one = prediction[0][1] 
        two = prediction[0][2]
        three = prediction[0][3]
        four = prediction[0][4]
        if zero == 1: pred = 'N - Sinus Rhythm. No Arrhythmia found, enjoy your day!'
        if one == 1: pred = 'S - Superventricular Premature. You should see a doctor soon.'
        if two == 1: pred = 'V - Ventricular Premature. You should see a doctor soon.'
        if three == 1: pred = 'F - Ventricular Fusion. You should see a doctor soon.'
        if four == 1: pred = 'Q - Unclassifiable. You may want to go see a doctor as the model was not able to classify the beat.'
        return pred

    if st.button("Predict Arrythmia from ECG"):
        st.header('The predicted arrhythmia is: ')
        st.header(model_prediction(features_df))

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
import matplotlib.pyplot as plt

st.title("Arrhythmia Classification from Electrocardiograms using Long-Short Term Memory Networks")
st.header("This app will allow you to choose a testing input (single beat CSV file) and it will output an arrythmia classification.")
    
uploaded_file = st.file_uploader("Choose the input file for classification.")


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
        X_stream = features_df.to_numpy()
        X_stream = np.reshape(X_stream, (X_stream.shape[0], 1, X_stream.shape[1]))
        X_stream_norm = (X_stream - 0.17428295254906354)/0.22632738719052128
        
        model = tf.keras.models.load_model("streamlit_model.hdf5")
        prediction = model.predict(X_stream_norm, verbose=1)
        prediction = (np.rint(prediction)).astype('int')
        zero = prediction[0][0] 
        one = prediction[0][1] 
        two = prediction[0][2]
        three = prediction[0][3]
        four = prediction[0][4]
        if zero == 1: pred = 'N - Sinus Rhythm. No Arrhythmia found, enjoy your day!'
        if one == 1: pred = 'S - Superventricular Premature. You should see a doctor immediately.'
        if two == 1: pred = 'V - Ventricular Premature. You should see a doctor immediately.'
        if three == 1: pred = 'F - Ventricular Fusion. You should see a doctor immediately.'
        if four == 1: pred = 'Q - Unclassifiable. The model determined the beat was unclassifiable, you may want to see a doctor soon..'
        return pred

    if st.button("Predict Arrythmia from ECG"):
        st.header('The predicted arrhythmia is: ')
        st.header(model_prediction(features_df))

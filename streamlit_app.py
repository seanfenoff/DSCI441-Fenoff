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

st.write("Electrocardiogram Arrythmia Detection using Long-Short Term Memory Networks")
st.write("This app will allow you to choose a testing input and it will output an arrythmia classification")

# train = pd.read_csv("mitbih_train.csv", header=None)
# train_y = train.iloc[:, -1]
# train_y = train_y.astype('int')
# train_x = train.iloc[:, :-1]
# train_x = train_x.astype('float')
# unique, counts = np.unique(train_y, return_counts=True)
# print(f'unique values: {unique}')
# print(f'counts: {counts}')
# class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

if st.checkbox('Show Example plot of Sinus Rythm - N'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(data=chart_data)

if st.checkbox('Show Example plot of Superventricular Premature - S'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(data=chart_data)

if st.checkbox('Show Example plot of Ventricular Premature - V'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(data=chart_data)
    
if st.checkbox('Show Example plot of Ventricular Fusion - F'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(data=chart_data)

if st.checkbox('Show Example plot of Unclassifiable - Q'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(data=chart_data)
    
uploaded_file = st.file_uploader("Choose input file for classification.")

test_df = pd.read_csv(uploaded_file, header=None)
# features_df = test_df.iloc[:,:-1]
# features_df = features_df.astype('float32')
# X_test_np = features_df.to_numpy()
# X_test_np = np.reshape(X_test_np, (X_test_np.shape[0], 1, X_test_np.shape[1]))

model = tf.saved_model.load('/Users/smfen/Documents/Lehigh Graduate School/Lehigh Spring 2023/DSCI 441 -- Stat. and ML/Project/streamlit_model')
model.predict(test_df, verbose=1)

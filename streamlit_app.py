#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:11:33 2023

@author: smfen
"""

import streamlit as st
import pandas as pd
import numpy as np

st.write("Electrocardiogram Arrythmia Detection using Long-Short Term Memory Networks")
st.write("This app will allow you to choose a testing input and it will output an arrythmia classification")

train = pd.read_csv("mitbih_train.csv", header=None)
train_y = train.iloc[:, -1]
train_y = train_y.astype('int')
train_x = train.iloc[:, :-1]
train_x = train_x.astype('float')
unique, counts = np.unique(train_y, return_counts=True)
print(f'unique values: {unique}')
print(f'counts: {counts}')
class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

if st.checkbox('Show Example Sinus Rythm - N'):
    chart_data = train_x[1]
    chart_data
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:12:54 2023

@author: smfen
"""

# DSCI 441 Project

import pandas as pd
import numpy as np

# First upload the data set into Python
data = pd.read_csv("/Users/smfen/Downloads/rais_anonymized 2/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv")

# See preliminary dataframe size
print(data.size)

# Clean some of it up
to_drop = ['id',
           'date',
           'spo2',
           'sleep_points_percentage',
           'daily_temperature_variation',
           'badgeType', 
           'mindfulness_session', 
           'scl_avg', 
           'minutesToFallAsleep', 
           'minutesAfterWakeup', 
           'minutes_in_default_zone_1', 
           'minutes_below_default_zone_1', 
           'minutes_in_default_zone_2', 
           'minutes_in_default_zone_3', 
           'step_goal', 
           'min_goal', 
           'max_goal', 
           'step_goal_label', 
           'ALERT', 
           'HAPPY', 
           'NEUTRAL', 
           'RESTED/RELAXED', 
           'SAD', 
           'TENSE/ANXIOUS', 
           'TIRED', 
           'ENTERTAINMENT', 
           'GYM', 
           'HOME', 
           'HOME_OFFICE', 
           'OTHER', 
           'OUTDOORS', 
           'TRANSIT', 
           'WORK/SCHOOL']

data.drop(to_drop, inplace=True, axis=1)




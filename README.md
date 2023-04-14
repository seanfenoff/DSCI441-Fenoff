# DSCI441-Fenoff
Repository for Lehigh University DSCI 441 project. 

This project is being used to classify Electrocardiogram (ECG) signals
given an input beat. The novel idea in the code is using Long-Short 
Term Memory (LSTM) nets in addition with confirmation of a more traditional
convolution neural net (CNN).

The first step was some data standardization, as the data points are in time
in 180ms section with millivolt (mV) readings. After this, the data was broken 
into training and testing sets and some exploratory signal analysis was complete. 

Then a CNN was trained and evaluated, and then the same for the LSTM model. 

Eventually I plan to used this code in a StreamLit application, so an example 
ECG could be input and the arrhythmia classification would output.  

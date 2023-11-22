import numpy as np 
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def oui():
    print("oui")
## write a function that returns the dataset.csv file as a pandas dataframe
import pandas as pd

def get_db():
    db = pd.read_csv('dataset.csv', sep=',')
    return db

def get_db2():
    db = pd.read_csv('tcc_ceds_music.csv', sep=',')
    return db
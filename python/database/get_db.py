## write a function that returns the dataset.csv file as a pandas dataframe
import pandas as pd

def get_db():
    db = pd.read_csv('csv/dataset.csv', sep=',')
    return db

def get_db2():
    db = pd.read_csv('csv/tcc_ceds_music.csv', sep=',')
    return db

def get_db3():
    db = pd.read_csv('csv/spotify_song.csv', sep=',')
    return db
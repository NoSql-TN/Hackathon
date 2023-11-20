## write a function that returns the dataset.csv file as a pandas dataframe
import pandas as pd

def get_db():
    db = pd.read_csv('dataset.csv', sep=',')
    return db
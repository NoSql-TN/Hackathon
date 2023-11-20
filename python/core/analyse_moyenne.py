from python.database.get_db import get_db

def get_moyenne_pop(file):
    return file['popularity'].mean()

def mean_pop_per_genre(file):
    return file.groupby('track_genre')['popularity'].mean()
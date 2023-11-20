from python.database.get_db import get_db

def get_totale_pop(file):
    return file['popularity'].mean()

def totale_pop_per_genre(file):
    return file.groupby('track_genre')['popularity'].mean()
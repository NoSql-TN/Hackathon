from python.database.get_db import get_db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from python.database.get_db import get_db

def get_all_graphs():
    db = get_db()
    df = pd.DataFrame(db)
    df = df.dropna()
    df = df.drop_duplicates()
    data = get_data_for_graphs(df)
    return data

def get_data_for_graphs(df):
    
    # the goal is to generate the data for all the graphs:
    # Histogramme popularité, histogramme duration,
    # camembert explicit
    # hsitogramme danceability
    # camembert key
    # graph energy
    # graph loudness
    # camembert speech
    # graph instru
    # camembert liveness
    # camembert genre
    # graph valence
    # tempo histogramme
    data={}
    data['popularity'] = get_popularity_data(df)
    data['duration'] = get_duration_data(df)
    data['explicit'] = get_explicit_data(df)
    data['danceability'] = get_danceability_data(df)
    data['key'] = get_key_data(df)
    data['energy'] = get_energy_data(df)
    data['loudness'] = get_loudness_data(df)
    data['speechiness'] = get_speechiness_data(df)
    data['instrumentalness'] = get_instrumentalness_data(df)
    data['liveness'] = get_liveness_data(df)
    data['genre'] = get_genre_data(df)
    data['valence'] = get_valence_data(df)
    data['tempo'] = get_tempo_data(df)
    return data
    
    
def get_popularity_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['popularity'], ascending=False)
    bins = np.linspace(0, 100, 50)
    bins = bins.astype(int)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['popularity'] >= bins[i]) & (df['popularity'] < bins[i+1])])
    return data

    
def get_duration_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['duration_ms'], ascending=False)
    df['duration_ms'] = df['duration_ms']/60000
    bins = np.linspace(0, 20, 50)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['duration_ms'] >= bins[i]) & (df['duration_ms'] < bins[i+1])])
    return data

def get_explicit_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['explicit'], ascending=False)
    explicit = df[df['explicit'] == 1]
    non_explicit = df[df['explicit'] == 0]
    data = {}
    data['Explicit'] = len(explicit)
    data['Non-Explicit'] = len(non_explicit)
    return data

def get_danceability_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['danceability'], ascending=False)
    bins = np.linspace(0, 1, 50)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['danceability'] >= bins[i]) & (df['danceability'] < bins[i+1])])
    return data

def get_key_data(df):
    
    # 0 = C, 1 = C#, 2 = D, 3 = D#, 4 = E, 5 = F, 6 = F#, 7 = G, 8 = G#, 9 = A, 10 = A#, 11 = B
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['key'], ascending=False)
    C = df[df['key'] == 0]
    C_sharp = df[df['key'] == 1]
    D = df[df['key'] == 2]
    D_sharp = df[df['key'] == 3]
    E = df[df['key'] == 4]
    F = df[df['key'] == 5]
    F_sharp = df[df['key'] == 6]
    G = df[df['key'] == 7]
    G_sharp = df[df['key'] == 8]
    A = df[df['key'] == 9]
    A_sharp = df[df['key'] == 10]
    B = df[df['key'] == 11]
    data = {}
    data['C'] = len(C)
    data['C#'] = len(C_sharp)
    data['D'] = len(D)
    data['D#'] = len(D_sharp)
    data['E'] = len(E)
    data['F'] = len(F)
    data['F#'] = len(F_sharp)
    data['G'] = len(G)
    data['G#'] = len(G_sharp)
    data['A'] = len(A)
    data['A#'] = len(A_sharp)
    data['B'] = len(B)
    return data

def get_energy_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['energy'], ascending=False)
    bins = np.linspace(0, 1, 50)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['energy'] >= bins[i]) & (df['energy'] < bins[i+1])])
    return data
    
def get_loudness_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['loudness'], ascending=False)
    bins = np.linspace(-50, 5, 56)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['loudness'] >= bins[i]) & (df['loudness'] < bins[i+1])])
    return data    

def get_speechiness_data(df):
    
    #Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'tempo', 'valence'])
    df = df.sort_values(by=['speechiness'], ascending=False)
    # There is 3 categories: 0.33-0.66, 0.66-1, 0-0.33
    data = {}
    data['both'] = len(df[(df['speechiness'] >= 0.33) & (df['speechiness'] < 0.66)])
    data['speech'] = len(df[(df['speechiness'] >= 0.66) & (df['speechiness'] < 1)])
    data['music'] = len(df[(df['speechiness'] >= 0) & (df['speechiness'] < 0.33)])
    return data

def get_instrumentalness_data(df):
    
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['instrumentalness'], ascending=False)
    bins = np.linspace(0, 1, 50)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['instrumentalness'] >= bins[i]) & (df['instrumentalness'] < bins[i+1])])
    return data

def get_liveness_data(df):
    
    # if above 0.8, it is probably a live song
    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence'])
    df = df.sort_values(by=['liveness'], ascending=False)
    data = {}
    data['live'] = len(df[df['liveness'] >= 0.8])
    data['not live'] = len(df[df['liveness'] < 0.8])
    return data

def get_genre_data(df):

    # get all the different genres in the dataframe by removing duplicates
    genres = df.track_genre.unique().tolist()
    # count the number of songs per genre
    data = {}
    for i in range(len(genres)):
        data[genres[i]] = len(df[df['track_genre'] == genres[i]])
    return data

def get_valence_data(df):

    df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'])
    df = df.sort_values(by=['valence'], ascending=False)
    bins = np.linspace(0, 1, 50)
    data = {}
    for i in range(len(bins)-1):
        data[str(bins[i])] = len(df[(df['valence'] >= bins[i]) & (df['valence'] < bins[i+1])])
    return data

def get_tempo_data(df):
    
        df = df.drop(columns=['Unnamed: 0', 'artists', 'popularity', 'duration_ms', 'explicit', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence'])
        df = df.sort_values(by=['tempo'], ascending=False)
        bins = np.linspace(0, 250, 50)
        data = {}
        for i in range(len(bins)-1):
            data[str(bins[i])] = len(df[(df['tempo'] >= bins[i]) & (df['tempo'] < bins[i+1])])
        return data

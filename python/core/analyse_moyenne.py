import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)


from python.database.get_db import get_db, get_db2

db = get_db()
file = pd.DataFrame(db)
file = file.dropna()
file = file.drop_duplicates()
    
db2 = get_db2()
file2 = pd.DataFrame(db2)
file2 = file2.dropna()
file2 = file2.drop_duplicates()
    
def get_stats2(artist,genre):
    stats = []
    
    if artist == False:
        popularity = get_popularity(file,False,genre)
        top_5_most_used_lyrics = get_filtered_lyrics(file2,False,genre)
        duration = get_duration(file,False,genre)
        explicit = get_explicit(file,False,genre)
        danceability = get_danceability(file,False,genre)
        key = get_key(file,False,genre)
        energy = get_energy(file,False,genre)
        loudness = get_loudness(file,False,genre)
        speechiness = get_speechiness(file,False,genre)
        instrumentalness = get_instrumentalness(file,False,genre)
        tempo = get_tempo(file,False,genre)
        most_pop_artists = get_most_pop_artists(file,genre)
        
        stats.append(popularity)
        stats.append(top_5_most_used_lyrics)
        stats.append(duration)
        stats.append(explicit)
        stats.append(danceability)
        stats.append(key)
        stats.append(energy)
        stats.append(loudness)
        stats.append(speechiness)
        stats.append(instrumentalness)
        stats.append(tempo)
        stats.append(most_pop_artists)
    elif genre == False:

        popularity = get_popularity(file,artist,False)
        top_5_most_used_lyrics = get_filtered_lyrics(file2,artist,False)
        duration = get_duration(file,artist,False)
        explicit = get_explicit(file,artist,False)
        danceability = get_danceability(file,artist,False)
        key = get_key(file,artist,False)
        energy = get_energy(file,artist,False)
        loudness = get_loudness(file,artist,False)
        speechiness = get_speechiness(file,artist,False)
        instrumentalness = get_instrumentalness(file,artist,False)
        tempo = get_tempo(file,artist,False)
        most_pop_genres = get_most_pop_genres(file,artist)
        
        stats.append(popularity)
        stats.append(top_5_most_used_lyrics)
        stats.append(duration)
        stats.append(explicit)
        stats.append(danceability)
        stats.append(key)
        stats.append(energy)
        stats.append(loudness)
        stats.append(speechiness)
        stats.append(instrumentalness)
        stats.append(tempo)
        stats.append(most_pop_genres)
    return stats
        
def get_stats():
    stats = []
    
    popularity = get_popularity(file,False,False) #0
    top_5_most_used_lyrics = get_filtered_lyrics(file2,False,False) #1
    duration = get_duration(file,False,False) #2
    explicit = get_explicit(file,False,False) #3
    danceability = get_danceability(file,False,False) #4
    key = get_key(file,False,False) #5
    energy = get_energy(file,False,False) #6
    loudness = get_loudness(file,False,False) #7
    speechiness = get_speechiness(file,False,False) #8
    instrumentalness = get_instrumentalness(file,False,False) #9
    tempo = get_tempo(file,False,False) #10
    most_pop_artists = get_most_pop_artists(file,False) #11
    most_pop_genres = get_most_pop_genres(file,False) #12
    
    stats.append(popularity)
    stats.append(top_5_most_used_lyrics)
    stats.append(duration)
    stats.append(explicit)
    stats.append(danceability)
    stats.append(key)
    stats.append(energy)
    stats.append(loudness)
    stats.append(speechiness)
    stats.append(instrumentalness)
    stats.append(tempo)
    stats.append(most_pop_artists)
    stats.append(most_pop_genres)
    return stats

def get_most_pop_artists(file,genre):
    if genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file
    # aggregate the popularity of each artist
    file = file.groupby(['artists']).agg({'popularity': 'mean'}).reset_index()
    file["popularity"] = file["popularity"].astype(int)
    file = file.sort_values(by=['popularity'], ascending=False)
    

    unique_artists = file['artists'].unique()
    for i in range(len(unique_artists)):
        unique_artists[i] = unique_artists[i].replace(";"," & ")
    top_3_artists = unique_artists[:3] if len(unique_artists) >= 3 else unique_artists
    top_3_artists_list = top_3_artists.tolist()
    return top_3_artists_list
    
def get_most_pop_genres(file,artist):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    else:
        file = file
    file["popularity"] = file["popularity"].astype(int)
    file = file.sort_values(by=['popularity'], ascending=False)
    
    unique_genres = file["track_genre"].unique()
    top_3_genres = unique_genres[:3] if len(unique_genres) >= 3 else unique_genres
    top_3_genres_list = top_3_genres.tolist()
    return top_3_genres_list

def get_popularity(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    popularity = file["popularity"].mean()
    popolarity = round(popularity,2)
    return popolarity

def get_duration(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    duration = file["duration_ms"].mean()
    duration = round(duration,0)
    
    duration_s = duration/1000
    duration_m = duration_s//60
    duration_m_s = duration_s%60
    if duration_s > 0:
        duration_s = int(duration_s)
        duration_m = int(duration_m)
        duration_m_s = int(duration_m_s)
        return [duration_s,duration_m,duration_m_s]
    else:
        return [0,0,0]

def get_danceability(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    danceability = file["danceability"].mean()*100
    danceability = round(danceability,1)
    return danceability

def get_key(file,artist,genre):
    # 0 = C, 1 = C#, 2 = D, 3 = D#, 4 = E, 5 = F, 6 = F#, 7 = G, 8 = G#, 9 = A, 10 = A#, 11 = B
    dico = {0:"C",1:"C#",2:"D",3:"D#",4:"E",5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    key = file["key"].mean()
    if key >= 0: 
        key = round(key,0)
        key = dico[key]
    else:
        key = "N/A"
    return key

def get_energy(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    energy = file["energy"].mean()*100
    energy = round(energy,1)
    return energy

def get_loudness(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    loudness = file["loudness"].mean()
    loudness = round(loudness,2)
    return loudness

def get_speechiness(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    speechiness = file["speechiness"].mean()*100
    speechiness = round(speechiness,1)
    return speechiness

def get_instrumentalness(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file
    instrumentalness = file["instrumentalness"].mean()*100
    instrumentalness = round(instrumentalness,1)
    return instrumentalness

def get_tempo(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else: 
        file = file
    tempo = file["tempo"].mean()
    if tempo >= 0:
        tempo = round(tempo,0)
        tempo = int(tempo)
    else:
        tempo = "N/A"
    return tempo

def get_filtered_lyrics(file2,artist,genre):
    if artist:
        filtered_lyrics = file2[file2['artist_name'].str.contains(artist, case=False)]['lyrics']
    elif genre:
        filtered_lyrics = file2[file2['genre'] == genre]['lyrics']
    else:
        filtered_lyrics = file2['lyrics']
    
    all_lyrics = ' '.join(filtered_lyrics)
    words = nltk.word_tokenize(all_lyrics)
    
    spanish_stop_words = set(stopwords.words('spanish'))
    stop_words = set(stopwords.words('english')).union(spanish_stop_words)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    word_counts = Counter(filtered_words)
    top_5_words = word_counts.most_common(5)
    
    return top_5_words

def get_explicit(file,artist,genre):
    if artist:
        file = file[file['artists'].str.contains(artist, case=False)]
    elif genre:
        file = file[file['track_genre'] == genre]
    else:
        file = file 
    file["explicit"] = file["explicit"].astype(int)
    explicit = file[file["explicit"] == 1]
    
    len_file = len(file)
    if len_file == 0:
        explicit_rate = 0
    else:
        explicit_rate = (len(explicit)/len_file)*100
        explicit_rate = round(explicit_rate,2)
    return explicit_rate
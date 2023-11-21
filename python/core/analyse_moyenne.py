import nltk
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')

from python.database.get_db import get_db, get_db2

def get_stats(artist,genre):
    if artist == "":
        popularity = get_
        
def get_stats():
    stats = []
    file = get_db()
    file2 = get_db2()
    
    popularity = file["popularity"].mean()
    top_5_most_used_lyrics = get_filtered_lyrics(file2,False,False)
    duration = file["duration_ms"].mean()
    explicit = get_explicit()
    danceability = file["danceability"].mean()
    key = file["key"].mean()
    energy = file["energy"].mean()
    loudness = file["loudness"].mean()
    speechiness = file["speechiness"].mean()
    instrumentalness = file["instrumentalness"].mean()
    liveness = file["liveness"].mean()

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
    stats.append(liveness)
    return stats

def get_filtered_lyrics(file2,artist,genre):
    if artist:
        filtered_lyrics = file2[file2['artist'] == artist]['lyrics']
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

def get_explicit():
    file = get_db()
    file["explicit"] = file["explicit"].astype(int)
    explicit = file["explicit"].sum()/len(file["explicit"])
    return explicit
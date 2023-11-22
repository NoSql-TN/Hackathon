import pickle
from python.utils.Generate_encoded_lyrics import preprocess_text
import numpy as np
import pandas as pd
from keras.models import load_model

def predict_popularity(data):
    lyrics = data['lyrics']
    genre = data['genre']
    artist_name = data['artist_name']
    track_name = data['track_name']
    tempo = data['tempo']
    key = data['cle']
    explicit = data['explicit']
    duration = data['duration']

    # first we need to get the encoded lyrics, the genre, the artist_name and the track_name
    
    # all the encoders are loaded from the pickle files
    lyrics_encoder = pickle.load(open('models/lyrics_encoder.pkl', 'rb'))
    genre_encoder = pickle.load(open('models/genre_encoder.pkl', 'rb'))
    artist_name_encoder = pickle.load(open('models/artist_name_encoder.pkl', 'rb'))
    track_name_encoder = pickle.load(open('models/track_name_encoder.pkl', 'rb'))
    
    # we need to preprocess the lyrics, the genre, the artist_name and the track_name
    preprocessed_lyrics =  preprocess_text(lyrics)
    preprocessed_genre = preprocess_text(genre)
    preprocessed_artist_name = preprocess_text(artist_name)
    preprocessed_track_name = preprocess_text(track_name)
    
    # Encoding the lyrics
    lyrics_encoded = lyrics_encoder.transform([preprocessed_lyrics]).toarray()
    feature_array = np.array(lyrics_encoder.get_feature_names_out())
    tfidf_sorting = np.argsort(lyrics_encoder.idf_)[::-1]
    top_n = feature_array[tfidf_sorting][:1000]

    lyrics_encoded = pd.DataFrame(lyrics_encoded, columns=lyrics_encoder.get_feature_names_out())
    lyrics_encoded = lyrics_encoded[top_n]

    # Encoding the genre
    genre_encoded = genre_encoder.transform([[preprocessed_genre]]).toarray()
    genre_encoded = pd.DataFrame(genre_encoded, columns=genre_encoder.categories_)
    
    # Encoding the artist_name
    artist_name_encoded = artist_name_encoder.transform([[preprocessed_artist_name]]).toarray()
    artist_name_encoded = pd.DataFrame(artist_name_encoded, columns=artist_name_encoder.categories_)
    artist_name_encoded = artist_name_encoded[artist_name_encoded.sum().sort_values(ascending=False).index]
    artist_name_encoded = artist_name_encoded.iloc[:, :100]
    
    # Encoding the track_name
    track_name_encoded = track_name_encoder.transform([preprocessed_track_name]).toarray()
    track_name_encoded = pd.DataFrame(track_name_encoded, columns=track_name_encoder.get_feature_names_out())
    track_name_encoded = track_name_encoded[track_name_encoded.sum().sort_values(ascending=False).index]
    track_name_encoded = track_name_encoded.iloc[:, :500]
    
    lyrics_length = len(preprocessed_lyrics.split())
    lyrics_length = pd.DataFrame([lyrics_length], columns=['lyrics_length'])
    
    
    model = load_model('models/from_lyrics.keras')
    
    X = pd.concat([genre_encoded, artist_name_encoded, track_name_encoded, lyrics_encoded, lyrics_length], axis=1)

    prediction = model.predict(X)
    # extract the 4 predictions
    danceability = pd.DataFrame([prediction[0][0]], columns=['danceability'])
    energy = pd.DataFrame([prediction[0][1]], columns=['energy'])
    acousticness = pd.DataFrame([prediction[0][2]], columns=['acousticness'])
    instrumentalness = pd.DataFrame([prediction[0][3]], columns=['instrumentalness'])
    
    # importat the new encoder
    genre_encoder_pop = pickle.load(open('models/genre_encoder_pop.pkl', 'rb'))
    artist_name_encoder_pop = pickle.load(open('models/artists_encoder_pop.pkl', 'rb'))
    
    genre_encoded_pop = genre_encoder_pop.transform([[genre]]).toarray()
    genre_encoded_pop = pd.DataFrame(genre_encoded_pop, columns=genre_encoder_pop.categories_)
    
    artist_name_encoded_pop = artist_name_encoder_pop.transform([[artist_name]]).toarray()
    artist_name_encoded_pop = pd.DataFrame(artist_name_encoded_pop, columns=artist_name_encoder_pop.categories_)
    
    duration = pd.DataFrame([duration], columns=['duration_ms'])
    explicit = pd.DataFrame([explicit], columns=['explicit'])
    explicit = explicit.astype(int)
    key = pd.DataFrame([key], columns=['key'])
    tempo = pd.DataFrame([tempo], columns=['tempo'])
    
    # print the types of every variable
    print(danceability.dtypes)
    print(energy.dtypes)
    print(acousticness.dtypes)
    print(instrumentalness.dtypes)
    print(tempo.dtypes)
    print(duration.dtypes)
    print(explicit.dtypes)
    print(key.dtypes)
    print(genre_encoded_pop.dtypes)
    print(artist_name_encoded_pop.dtypes)
    
    
    # concatenate the encoded columns with the other columns, explicit, key, tempo, duration
    X = pd.concat([danceability, energy, acousticness, instrumentalness, tempo, duration, explicit, key, genre_encoded_pop, artist_name_encoded_pop], axis=1)
    
    # load the model with pickle
    loaded_model = pickle.load(open('models/finalized_model.sav', 'rb'))
    
    # predict the popularity
    X=X.astype(float)
    X.columns = X.columns.astype(str)

    prediction = loaded_model.predict(X)
    
    return prediction[0]
    

    
    
    
    
    
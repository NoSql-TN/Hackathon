import pandas as pd 
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import  OneHotEncoder
import pickle


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)
    

if __name__ == "__main__":
    # Load your dataset
    # Assuming your dataset is in a CSV file named 'your_dataset.csv'
    data = pd.read_csv('csv/spotify_song.csv')
    data = data.drop('id', axis=1)


    # Assuming your dataframe is named 'df'

    # Keep as input only those columns: artist_name,track_name,genre,lyrics,len
    X = data[['artist_name','track_name','genre','lyrics','len']]
    X = X.astype({'artist_name': str,'track_name': str,'genre': str,'lyrics': str})

    # Keep as output only the column danceability, energy,accousticness,instrumentalness
    y = data[['danceability','energy','acousticness','instrumentalness']]



    # encode the genre column
    genre_encoder = OneHotEncoder(handle_unknown='ignore')
    genre_encoder.fit(X[['genre']])
    # save the encoder in a pickle file
    pickle.dump(genre_encoder, open('genre_encoder.pkl', 'wb'))
    genre_encoded = genre_encoder.transform(X[['genre']]).toarray()
    genre_encoded = pd.DataFrame(genre_encoded, columns=genre_encoder.categories_)
    # print the encoded genre column
    print("Genre encoded")

    # encode the artist_name column
    artist_name_encoder =  OneHotEncoder(handle_unknown='ignore')
    artist_name_encoder.fit(X[['artist_name']])
    pickle.dump(artist_name_encoder, open('artist_name_encoder.pkl', 'wb'))
    artist_name_encoded = artist_name_encoder.transform(X[['artist_name']]).toarray()
    artist_name_encoded = pd.DataFrame(artist_name_encoded, columns=artist_name_encoder.categories_)
    # keep only the 100 most important artists
    artist_name_encoded = artist_name_encoded[artist_name_encoded.sum().sort_values(ascending=False).index]
    artist_name_encoded = artist_name_encoded.iloc[:, :100]

    print("Artist name encoded")

    #Apply preprocessing to each element of the track_name column
    preprocessed_track_name = X['track_name'].apply(preprocess_text)
    preprocessed_track_name = preprocessed_track_name.astype(str)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_track_name)
    pickle.dump(vectorizer, open('track_name_encoder.pkl', 'wb'))
    track_name_encoded = tfidf_matrix.toarray()
    track_name_encoded = pd.DataFrame(track_name_encoded, columns=vectorizer.get_feature_names_out())
    # keep only the 500 most important words
    track_name_encoded = track_name_encoded[track_name_encoded.sum().sort_values(ascending=False).index]
    track_name_encoded = track_name_encoded.iloc[:, :500]


    # print("Track name encoded")
    
    #ignore all the rows with None as lyrics
    
    # data = data[data['parole'].notna()]
    # print(len(data))
    # data['parole'] = data['parole'].astype(str)
    preprocessed_lyrics =  data['parole'].apply(preprocess_text)
    # convert if needed the preprocessed_lyrics column to a string type
    preprocessed_lyrics = preprocessed_lyrics.astype(str)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit(preprocessed_lyrics)
    pickle.dump(vectorizer, open('lyrics_encoder.pkl', 'wb'))
    print("Lyrics vectorizer")
    # keep only the 1500 most important words
    top_n = tfidf_matrix.sum(axis=0).tolist()[0]
    top_n = np.array(top_n).argsort()[-1500:][::-1]
    top_n = np.array(vectorizer.get_feature_names_out())[top_n]
    
    

    print("Top n")
    print(top_n)
    
    # put the 1000 words in a .txt
    with open('top_n_words.txt', 'w') as f:
        for item in top_n:
            f.write('"%s",' % item)

    lyrics_encoded = tfidf_matrix.toarray()
    lyrics_encoded = pd.DataFrame(lyrics_encoded, columns=vectorizer.get_feature_names_out())
    # keep only the 1000 most important words
    lyrics_encoded = lyrics_encoded[top_n]


    print("Lyrics encoded")

    # Concatenate all the encoded columns with len column
    X = pd.concat([genre_encoded, artist_name_encoded, track_name_encoded, lyrics_encoded, X['len']], axis=1)

    # save the encoded columns in a csv file
    X.to_csv('csv/encoded_columns.csv', index=False)

    print("X encoded")
    print(X.head())

#
from flask import Blueprint, request
from flask.templating import render_template
from python.core.prediction import predict_popularity

# Definition of the blueprint
predictionBP = Blueprint('predictionBP', __name__)

# Definition of the main route
@predictionBP.route("/prediction")
def prediction():
    genres = ['blues','country','hip hop','jazz','pop','reggae','rock']
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return render_template("prediction.html", genre=genres, note=notes)

@predictionBP.route("/get-result", methods=['POST'])    
def get_result():  
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    
    paroles = request.form['paroles']
    genre = request.form['genre']
    artist_name = request.form['artist_name']
    track_name = request.form['track_name']
    tempo = int(request.form['tempo'])
    key = notes.index(request.form['key'])
    explicit = int(request.form['explicit'])
    duration = int(request.form['duration'])*1000
    data = {"lyrics": paroles, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": key , "explicit" : explicit, "duration" : duration}
    
    score = predict_popularity(data)
    return render_template("result-pred.html",score=score)

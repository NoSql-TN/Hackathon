import os
from flask import Blueprint, request
from flask.templating import render_template
from python.core.prediction import predict_popularity
from python.core.amelioration import amelioration

# Definition of the blueprint
predictionBP = Blueprint('predictionBP', __name__)

# Definition of the main route
@predictionBP.route("/prediction")
def prediction():
    args = request.args
    if args:
        paroles = args['paroles']
        genre = args['genre']
        artist_name = args['artist_name']
        track_name = args['track_name']
        tempo = int(args['tempo'])
        key = int(args['key'])
        explicit = int(args['explicit'])
        duration = int(int(args['duration'])/1000)
    else:
        paroles, genre, artist_name, track_name, tempo, key, explicit, duration = "", "", "", "", 100, 0, 0, 140
    genres = ['blues','country','hip hop','jazz','pop','reggae','rock']
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return render_template("prediction.html", genres=genres, note=notes, score=-1, paroles=paroles, genre=genre, artist_name=artist_name, track_name=track_name, tempo=tempo, key=key, explicit=explicit, duration=duration)

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
    score = round(predict_popularity(data),2)
    with open('prediction.csv', 'w') as f:
        if os.stat('prediction.csv').st_size == 0:
            f.write('paroles,genre,artist_name,track_name,tempo,key,explicit,duration,score\n')
        f.write(f'{paroles},{genre},{artist_name},{track_name},{tempo},{key},{explicit},{duration},{score}\n')
    return render_template("prediction.html", score=score, genres=genre, note=notes, paroles=paroles, genre=genre, artist_name=artist_name, track_name=track_name, tempo=tempo, key=key, explicit=explicit, duration=duration)

@predictionBP.route("/get-amelioration", methods=['GET'])
def get_amelioration():
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    
    args = request.args
    paroles = args['paroles']
    genre = args['genre']
    artist_name = args['artist_name']
    track_name = args['track_name']
    tempo = int(args['tempo'])
    key = int(args['key'])
    explicit = int(args['explicit'])
    duration = int(int(args['duration'])/1000)
    score = float(args['score'])
    data = {"lyrics": paroles, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": key , "explicit" : explicit, "duration" : duration, "score" : score}
    improvement, alreadygood = amelioration(data)
    print("improvement")
    print(improvement)
    if improvement != {}:
        best_score = score
        for variable in improvement.keys():
            if improvement[variable][1] > best_score:
                best_score = improvement[variable][1]
    print("alreadygood")
    print(alreadygood)
    alreadygood = {}
    return render_template("amelioration.html", score=score, genres=genre, note=notes, paroles=paroles, genre=genre, artist_name=artist_name, track_name=track_name, tempo=tempo, key=key, explicit=explicit, duration=duration, diff=round(best_score-score,2), improvement=improvement, alreadygood=alreadygood)
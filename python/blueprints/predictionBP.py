from flask import Blueprint,session
from flask.templating import render_template
from python.core.prediction import predict_popularity

# Definition of the blueprint
predictionBP = Blueprint('predictionBP', __name__)

# Definition of the main route
@predictionBP.route("/prediction")
def prediction():
    #get data from the form     
    fake_data = {"lyrics": "I'm a little teapot, short and stout. Here is my handle, here is my spout. When I get all steamed up, hear me shout. Tip me over and pour me out!", "genre": "Pop", "artist_name": "Ariana Grande", "track_name": "tea pot", "tempo": 120, "clé": 0 , "explicit" : True, "duration" : 200}
    score = predict_popularity(fake_data)
    print(score)
    genres = ['blues','country','hip hop','jazz','pop','reggae','rock']
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return render_template("prediction.html", genre=genres, note=notes, score=score)

from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
predictionBP = Blueprint('predictionBP', __name__)

# Definition of the main route
@predictionBP.route("/prediction")
def prediction():
    genres = ['blues','country','hip hop','jazz','pop','reggae','rock']
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return render_template("prediction.html", genre=genres, note=notes)
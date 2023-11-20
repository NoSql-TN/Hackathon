from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
predictionBP = Blueprint('predictionBP', __name__)

# Definition of the main route
@predictionBP.route("/prediction")
def prediction():
    return render_template("prediction.html")
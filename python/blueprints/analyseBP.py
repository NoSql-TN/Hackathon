from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
analyseBP = Blueprint('analyseBP', __name__)    

from python.core.analyse_moyenne import get_moyenne_pop, mean_pop_per_genre, duration_popularity
from python.database.get_db import get_db


# Definition of the main route
@analyseBP.route("/analyse-moy")
def analyseMoy():
    print(get_moyenne_pop(get_db()))
    mean_pop_per_genre_fig = mean_pop_per_genre(get_db())
    dur_pop_fig = duration_popularity(get_db())
    return render_template("analyse.html", dur_pop_fig = dur_pop_fig, mean_pop_per_genre_fig = mean_pop_per_genre_fig)

# Definition of the main route
@analyseBP.route("/analyse-tot")
def analyseTot():
    return render_template("analyse.html")
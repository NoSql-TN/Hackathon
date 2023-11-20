from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
analyseBP = Blueprint('analyseBP', __name__)    

from python.core.analyse_moyenne import get_moyenne_pop, mean_pop_per_genre
from python.core.analyse_totale import get_totale_pop, totale_pop_per_genre
from python.database.get_db import get_db


# Definition of the main route
@analyseBP.route("/analyse-moy")
def analyseMoy():
    print(get_moyenne_pop(get_db()))
    print(mean_pop_per_genre(get_db()))
    return render_template("analyse-moy.html")

# Definition of the main route
@analyseBP.route("/analyse-tot")
def analyseTot():
    print(get_totale_pop(get_db()))
    print(totale_pop_per_genre(get_db()))
    return render_template("analyse-tot.html")
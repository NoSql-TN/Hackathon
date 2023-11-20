from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
analyseBP = Blueprint('analyseBP', __name__)    

from python.core.analyse_moyenne import get_moyenne_pop, mean_pop_per_genre
from python.database.get_db import get_db


# Definition of the main route
@analyseBP.route("/analyse")
def analyse():
    print(get_moyenne_pop(get_db()))
    print(mean_pop_per_genre(get_db()))
    return render_template("analyse.html")
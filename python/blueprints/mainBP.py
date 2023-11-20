from flask import Blueprint,session,redirect
from flask.templating import render_template

# Definition of the blueprint
mainBP = Blueprint('mainBP', __name__)

from python.core.analyse_moyenne import get_moyenne_pop, mean_pop_per_genre
from python.database.get_db import get_db

# Definition of the main route
@mainBP.route("/")
@mainBP.route("/home")
def home():
    return redirect("/generation")

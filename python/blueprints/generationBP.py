from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
generationBP = Blueprint('generationBP', __name__)

from python.core.lyrics_generator import generator

# Definition of the main route
@generationBP.route("/generation")
def generation():
    generator()
    return render_template("generation.html")

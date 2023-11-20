from flask import Blueprint,session
from flask.templating import render_template

# Definition of the blueprint
generationBP = Blueprint('generationBP', __name__)

# Definition of the main route
@generationBP.route("/generation")
def generation():
    return render_template("generation.html")

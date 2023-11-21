import json
from flask import Blueprint, request,session
from flask.templating import render_template

# Definition of the blueprint
analyseBP = Blueprint('analyseBP', __name__)    

# from python.core.analyse_moyenne import get_moyenne_pop, mean_pop_per_genre, duration_popularity
from python.core.analyse_totale import get_all_graphs, get_all_graphs_with_filter
from python.database.get_db import get_db


# Definition of the stats route
# @analyseBP.route("/analyse-stats")
# def analyseMoy():
    # print(get_moyenne_pop(get_db()))
    # mean_pop_per_genre_fig = mean_pop_per_genre(get_db())
    # dur_pop_fig = duration_popularity(get_db())
    # return render_template("analyse-stats.html", dur_pop_fig = dur_pop_fig, mean_pop_per_genre_fig = mean_pop_per_genre_fig)

# Definition of the graph route
@analyseBP.route("/analyse-graph", methods=['GET', 'POST'])
def analyseTot():
    # check if request is methods POST
    if request.method == 'POST':
        filter = request.form.get('filter')
        artiste = request.form.get('artiste') == None
        print(filter)
        print(artiste)
        if filter:
            data = get_all_graphs_with_filter(filter,artiste)
        else:
            data = get_all_graphs()
        with open("static/data.json", "w") as f:
            json.dump(data, f, indent=4)
        return render_template("analyse-graph.html", name=filter, artiste=artiste)
    else:
        data = get_all_graphs()
        with open("static/data.json", "w") as f:
            json.dump(data, f, indent=4)
        return render_template("analyse-graph.html", name="")

@analyseBP.route("/get-data-graph-popularity")
def get_data_graph():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('popularity')

@analyseBP.route("/get-data-graph-duration")
def get_data_graph_duration():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('duration')

@analyseBP.route("/get-data-graph-explicit")
def get_data_graph_explicit():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('explicit')

@analyseBP.route("/get-data-graph-danceability")
def get_data_graph_danceability():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('danceability')

@analyseBP.route("/get-data-graph-key")
def get_data_graph_key():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('key')

@analyseBP.route("/get-data-graph-energy")
def get_data_graph_energy():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('energy')

@analyseBP.route("/get-data-graph-loudness")
def get_data_graph_loudness():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('loudness')

@analyseBP.route("/get-data-graph-speechiness")
def get_data_graph_speechiness():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('speechiness')

@analyseBP.route("/get-data-graph-instrumentalness")
def get_data_graph_instrumentalness():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('instrumentalness')

@analyseBP.route("/get-data-graph-liveness")
def get_data_graph_liveness():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('liveness')

@analyseBP.route("/get-data-graph-genre")
def get_data_graph_genre():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('genre')

@analyseBP.route("/get-data-graph-valence")
def get_data_graph_valence():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('valence')

@analyseBP.route("/get-data-graph-tempo")
def get_data_graph_tempo():
    with open("static/data.json", "r") as f:
        data = json.load(f)
    return data.get('tempo')




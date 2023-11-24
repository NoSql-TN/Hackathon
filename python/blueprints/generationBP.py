from flask import Blueprint,session, request
from flask.templating import render_template
from python.core.prediction import predict_popularity_from_lyrics

import paramiko

# Definition of the blueprint
generationBP = Blueprint('generationBP', __name__)

# Definition of the main route
@generationBP.route("/generation", methods=["GET", "POST"])
def generation():
    if request.method == "POST":
        input_user = request.form.get("input")
        return render_template("loading.html", lyrics=input_user)
    elif request.method == "GET":
        return render_template("generation.html", lyrics="")

@generationBP.route("/generated")
def generated():
    input_user = ""
    inputito = request.args.to_dict(flat=False)['lyrics'][0]
    if inputito:
        input_user = str(inputito)
  
    host = 'tesla.telecomnancy.univ-lorraine.fr'
    username = 'bourdais4u'
    password = 'X.W.E5L3Rb'

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh_client.connect(hostname=host, username=username, password=password)
        print("Connected to SSH server")
        
        stdin, stdout, stderr = ssh_client.exec_command('cd PI/Remi/Hackathon2/;source venv/bin/activate;python3 python/core/lyrics_generator.py '+input_user+';cat lyrics.txt')

        output = stdout.read().decode('utf-8')
        lyrics_list1 = output.split("--sep--")
        lyrics_list = lyrics_list1[:-1]
        score_list = []
        for i in range(len(lyrics_list)):
            score_list.append(predict_popularity_from_lyrics(lyrics_list[i]))
        print(score_list)
        return render_template("generation.html", lyrics=lyrics_list, scores=score_list)

    except paramiko.AuthenticationException:
        print("Authentication failed, please check your credentials.")
    except paramiko.SSHException as ssh_exception:
        print("SSH connection failed:", ssh_exception)
    finally:
        ssh_client.close()
# Import neded packages
import os
from flask import Flask, redirect, flash, send_from_directory


# Definition of the app
def create_app() -> Flask:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = '29FTh4Swfr3DuMlNRcQcZxCk7IFBMooP'

    # Import blueprints
    from python.blueprints.mainBP import mainBP
    app.register_blueprint(mainBP)
    
    from python.blueprints.analyseBP import analyseBP
    app.register_blueprint(analyseBP)
    
    from python.blueprints.generationBP import generationBP
    app.register_blueprint(generationBP)
    
    from python.blueprints.predictionBP import predictionBP
    app.register_blueprint(predictionBP)

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'),
                            'favicon.ico',mimetype='image/vnd.microsoft.icon')
        
    # Error 404 handler
    @app.errorhandler(404)
    def pageNotFound(error):
        flash("HTTP 404 Not Found", "Red_flash")
        return redirect('/')

    return app


# Start app if file is not imported
if __name__ == "__main__":
    app = create_app()
    app.run(debug=1, host='0.0.0.0', port=5454)

from flask import Flask
from backend.Othello.get_engine_move import othello_move_blueprint
from backend.Othello.get_evaluation import othello_evaluation_blueprint

application = Flask(__name__)

application.register_blueprint(othello_move_blueprint)
application.register_blueprint(othello_evaluation_blueprint)

@application.route('/')
def index():
    return application.send_static_file('index.html')

if __name__ == '__main__':
    application.run(debug = True)
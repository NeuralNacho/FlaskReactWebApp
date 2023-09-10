from flask import Flask
from backend.GeographyQuiz.database_models import db, bcrypt, jwt

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///geography_quiz_database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = '`l0^4Fm>MR]}fI&2+nF;Amu=1!mHjq'

application = Flask(__name__)

application.config.from_object(Config)

@application.route('/')
def index():
    return application.send_static_file('index.html')

if __name__ == '__main__':
    from backend.Othello.get_engine_move import othello_move_blueprint
    from backend.Othello.get_evaluation import othello_evaluation_blueprint
    application.register_blueprint(othello_move_blueprint)
    application.register_blueprint(othello_evaluation_blueprint)

    from backend.GeographyQuiz.create_user import create_user_blueprint
    from backend.GeographyQuiz.login import login_blueprint
    from backend.GeographyQuiz.add_quiz_user_answer import add_quiz_user_answer_blueprint
    from backend.GeographyQuiz.get_quiz_user_history import get_quiz_user_history_blueprint
    from backend.GeographyQuiz.get_new_question import get_new_question_blueprint
    from backend.GeographyQuiz.get_random_starting_question import get_random_starting_question_blueprint
    application.register_blueprint(create_user_blueprint)
    application.register_blueprint(login_blueprint)
    application.register_blueprint(add_quiz_user_answer_blueprint)
    application.register_blueprint(get_quiz_user_history_blueprint)
    application.register_blueprint(get_new_question_blueprint)
    application.register_blueprint(get_random_starting_question_blueprint)
    db.init_app(application)
    bcrypt.init_app(application)
    jwt.init_app(application)
    with application.app_context():
        db.create_all()

    application.run(debug = True)
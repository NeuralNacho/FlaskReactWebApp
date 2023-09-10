from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from sqlalchemy import JSON

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()

class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(50), unique = True, nullable = False)
    password_hash = db.Column(db.String(60), nullable = False)
    quiz_user_data = db.Column(JSON) # JSON will be list of length 4 dictionaries

    def __init__(self, username, password_hash, quiz_user_data = None):
        self.username = username
        self.password_hash = password_hash
        self.quiz_user_data = quiz_user_data or [] # Is [] if None

class QuestionAndAnswer(db.Model):
    __tablename__ = 'question_and_answer'
    id = db.Column(db.Integer, primary_key = True)
    question = db.Column(db.String(500), unique = True, nullable = False)
    question_category = db.Column(db.String(50), nullable = False)
    answer = db.Column(db.String(200), nullable = False)
    answer_info = db.Column(db.String(1000), nullable = False)
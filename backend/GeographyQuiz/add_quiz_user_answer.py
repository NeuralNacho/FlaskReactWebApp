from flask import request, jsonify, Blueprint
from .database_models import User
from application import db
from flask_jwt_extended import jwt_required, get_jwt_identity
import json

add_quiz_user_answer_blueprint = Blueprint('add_quiz_user_answer_blueprint', __name__)

@add_quiz_user_answer_blueprint.route('/add_quiz_user_answer', methods = ['POST'])
@jwt_required()
def add_quiz_user_answer():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id = user_id).first()
    new_user_answer = request.get_json()
    existing_user_answers = json.loads(user.quiz_user_data) if user.quiz_user_data else []
    existing_user_answers.append(new_user_answer)
    quiz_data_json = json.dumps(existing_user_answers)
    user.quiz_user_data = quiz_data_json
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'Quiz user answer added successfully'})
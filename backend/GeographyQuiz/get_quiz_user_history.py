from flask import Blueprint, jsonify
from .database_models import User
from flask_jwt_extended import jwt_required, get_jwt_identity
import json

get_quiz_user_history_blueprint = Blueprint('get_quiz_user_history_blueprint', __name__)

@get_quiz_user_history_blueprint.route('/get_quiz_user_history', methods = ['GET'])
@jwt_required()
def get_quiz_user_history():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id = user_id).first()
    quiz_user_history = user.quiz_user_data

    # Ensure quiz_user_history is a valid JSON string and convert it to a dictionary
    try:
        # Turning the string to JSON
        quiz_user_history = json.loads(quiz_user_history)
    except json.JSONDecodeError:
        # Handle the case where quiz_user_history is not valid JSON
        return jsonify({"error": "Invalid JSON data"}), 400


    return jsonify(quiz_user_history)
from flask import request, jsonify, Blueprint
from application import bcrypt
from .database_models import User
from flask_jwt_extended import create_access_token

login_blueprint = Blueprint('login_blueprint', __name__)

@login_blueprint.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']
    user = User.query.filter_by(username = username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity = user.id)
        return jsonify({'message': 'Success', 'accessToken': access_token})
    else:
        return jsonify({'message': 'Failure'})
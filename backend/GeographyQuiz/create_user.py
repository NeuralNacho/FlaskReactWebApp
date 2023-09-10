from flask import request, jsonify, Blueprint
from sqlalchemy.exc import IntegrityError
from application import bcrypt, db
from .database_models import User
from flask_jwt_extended import create_access_token

create_user_blueprint = Blueprint('create_user_blueprint', __name__)

@create_user_blueprint.route('/create_user', methods = ['POST'])
def create_user():
    data = request.get_json()
    username = data['username']
    try:
        password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    except:
        return jsonify({'message': 'Password cannot be empty'})
    user = User(username = username, password_hash = password_hash)

    try:
        db.session.add(user)
        db.session.commit()
        access_token = create_access_token(identity = user.id)
        return jsonify({'message': 'Success', 'accessToken': access_token})
    except IntegrityError:
        db.session.rollback()  # Rollback the transaction
        return jsonify({'message': 'Username already exists'})
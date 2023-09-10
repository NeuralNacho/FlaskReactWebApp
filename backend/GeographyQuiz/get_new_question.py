from flask import request, jsonify, Blueprint
from .database_models import QuestionAndAnswer

get_new_question_blueprint = Blueprint('get_new_question_blueprint', __name__)

@get_new_question_blueprint.route('/get_new_question', methods = ['POST'])
def get_new_question():
    data = request.get_json()
    new_primary_key = data['new_primary_key']

    row_count = QuestionAndAnswer.query.count()

    new_primary_key = (new_primary_key % row_count) + 1
    print(new_primary_key)
    question_and_answer = QuestionAndAnswer.query.filter_by(id = new_primary_key).first()

    return jsonify({'primary_key': question_and_answer.id,
                    'question': question_and_answer.question,
                    'question_category': question_and_answer.question_category,
                    'answer': question_and_answer.answer,
                    'answer_info': question_and_answer.answer_info})

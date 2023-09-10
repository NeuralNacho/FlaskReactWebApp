from flask import jsonify, Blueprint
from .database_models import QuestionAndAnswer
import random

get_random_starting_question_blueprint = Blueprint('get_random_starting_question_blueprint', __name__)

@get_random_starting_question_blueprint.route('/get_random_starting_question', methods = ['POST'])
def get_random_starting_question():
    row_count = QuestionAndAnswer.query.count()

    print('RowCount', row_count)
    starting_primary_key = random.randint(1, row_count)
    question_and_answer = QuestionAndAnswer.query.filter_by(id = starting_primary_key).first()

    return jsonify({'primary_key': question_and_answer.id,
                    'question': question_and_answer.question,
                    'question_category': question_and_answer.question_category,
                    'answer': question_and_answer.answer,
                    'answer_info': question_and_answer.answer_info})

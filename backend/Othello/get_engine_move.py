from flask import request, jsonify
from flask import Blueprint
from .game_info_functions import *
from .neural_net_evaluation import *
from .heuristic_evaluation import *
from tensorflow import keras
from keras.models import load_model
import cProfile
from pstats import Stats

othello_move_blueprint = Blueprint('othello_move_blueprint', __name__)

@othello_move_blueprint.route('/get_engine_move', methods=['POST'])
def get_engine_move():
    if request.method == 'POST':
        # Get the data sent in the request's JSON body
        data = request.json
        board = data.get('board')
        current_player = data.get('current_player')
        engine_to_use = data.get('engine_to_use')
        game_state = get_game_state(board, current_player)
        
        start_time = time.time()

        if engine_to_use == 'Small Network':
            loaded_model = load_model('./backend/Othello/Neural_Net/depth7_small_model.keras')
            move, evaluation, number_of_nodes = perfect_end_search(game_state, loaded_model)
            evaluation *= 64
            evaluation -= 32
            evaluation = "{:.2f}".format(evaluation)
        elif engine_to_use == 'Large Network':
            loaded_model = load_model('./backend/Othello/Neural_Net/depth7_large_model.keras')
            move, evaluation, number_of_nodes = perfect_end_search(game_state, loaded_model)
            evaluation *= 64
            evaluation -= 32
            evaluation = "{:.2f}".format(evaluation)
        else:
            depth = int(engine_to_use[-1]) # e.g. Have something like engine_to_use = Non-AI Depth 5
            move, evaluation, number_of_nodes = iterative_deepening_search(game_state, depth)
        move = get_move_index(move)
        time_taken = time.time() - start_time
        print('Evaluation: ', evaluation)
        print(time_taken, '\n')
        return jsonify({"move": move, "evaluation": evaluation, "nodes": int(number_of_nodes), \
                        "timeTaken": time_taken})
    else:
        return jsonify({"error": "Invalid request method"})
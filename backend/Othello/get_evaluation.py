from flask import request, jsonify
from flask import Blueprint
from .game_info_functions import *
from .neural_net_evaluation import *
from .heuristic_evaluation import *
import time
from tensorflow import keras
from keras.models import load_model
import cProfile
from pstats import Stats

othello_evaluation_blueprint = Blueprint('othello_evaluation_blueprint', __name__)

@othello_evaluation_blueprint.route('/get_evaluation', methods=['POST'])
def get_evaluation():
    if request.method == 'POST':
        data = request.json
        board = data.get('board')
        current_player = data.get('current_player')
        engine_to_use = data.get('engine_to_use')
        game_state = get_game_state(board, current_player)
        
        start_time = time.time()

        if engine_to_use == 'Small Supervised AI':
            loaded_model = load_model('./backend/Othello/Neural_Net/depth7_small_model.keras')
            evaluations_board, number_of_nodes = evaluation_search(game_state, loaded_model)
        elif engine_to_use == 'Large Supervised AI':
            loaded_model = load_model('./backend/Othello/Neural_Net/depth7_large_model.keras')
            evaluations_board, number_of_nodes = evaluation_search(game_state, loaded_model)
        else:
            depth = int(engine_to_use[-1]) # e.g. Have something like engine_to_use = Non-AI Depth 5
            evaluations_board, number_of_nodes = iterative_deepening_evaluation_search(game_state, depth)

        time_taken = time.time() - start_time
        return jsonify({"evaluationsBoard": evaluations_board, "nodes": int(number_of_nodes), \
                        "timeTaken": time_taken})
    else:
        return jsonify({"error": "Invalid request method"})
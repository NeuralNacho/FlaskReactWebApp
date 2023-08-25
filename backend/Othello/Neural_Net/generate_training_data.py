import numpy as np
import random
import os
from ..game_info_functions import *
from ..get_engine_move import *

def rotate_bitboard(bitboard):
    # Rotates bitboard by 90 degrees clockwise
    rotated_bitboard = 0
    for row in range(8):
        for col in range(8):
            # See if there is a bit at (row, col) from the original bitboard
            bit = (bitboard >> (row * 8 + col)) & 1
            # Set the corresponding bit in the rotated bitboard
            rotated_bitboard |= bit << ((7 - col) * 8 + row)
    return rotated_bitboard

def reflect_bitboard(bitboard):
    # Reflects bitboard on vertical axis
    reflected_bitboard = 0
    for row in range(8):
        original_row = (bitboard >> (row * 8)) & 0xFF  # Extract the row from the original bitboard
        reflected_row = int(format(original_row, '08b')[::-1], 2)  # Reflect the bits within the row
        reflected_bitboard |= reflected_row << (row * 8)  # Set the reflected row in the new bitboard
    return reflected_bitboard

def add_data(game_state, position_array, evaluation_array, evaluation_depth):
    if evaluation_depth == 0:
        black_bitboard = game_state.black_bitboard
        white_bitboard = game_state.white_bitboard 
        current_player = game_state.current_player
        evaluation = heuristic_evaluation(game_state)
        game_state.current_player = current_player 
        # ^ Avoid issue with heuristic_eval changing current_player
    else: # i.e. evaluation depth > 0
        black_bitboard = game_state.black_bitboard
        white_bitboard = game_state.white_bitboard
        _, evaluation = iterative_deepening_search(game_state, evaluation_depth)
    
    if game_state.current_player == 2: 
        # Swap bitboards so that 'black_bitboard' in data is always bitboard of current player
        black_bitboard = game_state.white_bitboard
        white_bitboard = game_state.black_bitboard
        evaluation = -evaluation

    # heuristic_eval can give eval up to 70 but we restrict to this range for
    # when normalising the data
    if evaluation > 32:
        evaluation = 32
    if evaluation < -32:
        evaluation = -32

    for _ in range(2): # Two different reflections
        new_black_bitboard = reflect_bitboard(black_bitboard)
        new_white_bitboard = reflect_bitboard(white_bitboard)
        for _ in range(4): # Four different rotations for each reflection
            new_black_bitboard = rotate_bitboard(new_black_bitboard)
            new_white_bitboard = rotate_bitboard(new_white_bitboard)
            black_binary_string = format(new_black_bitboard, '064b')
            white_binary_string = format(new_white_bitboard, '064b')
            black_array = np.array([int(bit) for bit in black_binary_string], dtype = np.int8)
            white_array = np.array([int(bit) for bit in white_binary_string], dtype = np.int8)
            # Arrays given shape (8, 8, 1)
            black_array = black_array.reshape(8, 8, 1)
            white_array = white_array.reshape(8, 8, 1)
            position = np.dstack((black_array, white_array)) # Gets shape (8, 8, 2)
            position_array.append(position)
            evaluation_array.append(evaluation)
    return position_array, evaluation_array

def save_data(new_positions, new_evaluations, data_path):
    new_positions = np.array(new_positions)
    new_evaluations = np.array(new_evaluations)
    if os.path.exists(data_path):
        existing_data = np.load(data_path, allow_pickle = True)
        existing_positions = existing_data.item()['positions']
        existing_evaluations = existing_data.item()['evaluations']
        training_data_positions = np.append(existing_positions, new_positions, axis = 0)
        training_data_evaluations = np.append(existing_evaluations, new_evaluations)
    else:
        training_data_positions = new_positions
        training_data_evaluations = new_evaluations
    np.save(data_path, {'positions': training_data_positions, 'evaluations': training_data_evaluations})

def choose_random_move(game_state):
    legal_moves_bitboard = find_legal_moves(game_state)
    legal_moves_array = []
    while legal_moves_bitboard:
        move_bitboard = legal_moves_bitboard & -legal_moves_bitboard
        legal_moves_bitboard ^= move_bitboard
        legal_moves_array.append(move_bitboard)
    move = random.choice(legal_moves_array)
    return move

if __name__ == '__main__':
    # VARIABLES:
    SAVE_FILE_PATH = './backend/Othello/Neural_Net_Training/Training_Data/depth_0.npy'
    EVALUATION_DEPTH = 0
    MOVE_DEPTH = 0 # 0 for each player to make random moves

    black_start = (1 << 28) | (1 << 35)
    white_start = (1 << 27) | (1 << 36)

    training_data_positions = []
    training_data_evaluations = []
    save_no = 1
    while True:
        if len(training_data_evaluations) >= 50000:
            print('New save. Save no: ', save_no, 'Total evaluations added: ~', 50000 * save_no)
            save_no += 1
            save_data(training_data_positions, training_data_evaluations, SAVE_FILE_PATH)
            training_data_positions = []
            training_data_evaluations = []
        game_state = OthelloGameState(black_start, white_start, 1)
        
        # Get a rather random starting position for any MOVE_DEPTH
        for _ in range(6):
            random_move = choose_random_move(game_state)
            game_state = handle_legal_move(game_state, random_move)
            # Will not have data for starting position since move symmetric
            training_data_positions, training_data_evaluations = \
                add_data(game_state, training_data_positions, training_data_evaluations, EVALUATION_DEPTH)
        
        ply_no = 8
        two_turns_passed = False
        while not two_turns_passed:
            if MOVE_DEPTH == 0 or ply_no % 7 == 0:
                # This random move guaruntees variation in the data ontop of starting position
                move = choose_random_move(game_state)
            else:
                move, _ = iterative_deepening_search(game_state, MOVE_DEPTH)
            game_state = handle_legal_move(game_state, move)
            ply_no += 1
            if not find_legal_moves(game_state):
                opposite_player = 2 if game_state.current_player == 1 else 1
                game_state.current_player = opposite_player # Pass the turn
                if not find_legal_moves(game_state):
                    two_turns_passed = True
            # Pass turn before adding data (doesn't actually affect data quality)
            training_data_positions, training_data_evaluations = \
                add_data(game_state, training_data_positions, training_data_evaluations, EVALUATION_DEPTH)
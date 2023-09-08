''' In this file generate data for and train models which play through games 
and then learn from the final score of the game by fitting the positions 
observed to the final score'''


import numpy as np
import os
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn import preprocessing, model_selection
from .generate_training_data import *
from ..get_engine_move import *
from ..game_info_functions import *
from ..neural_net_evaluation import *

def get_model(model_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    else: # Make a new model
        # Large Model:
        # Create a sequential model
        model = Sequential()

        # Add Convolutional layers with residual connections
        # Note testing shows model is slower with batch normalisation
        # L2 regularisation layer can cause loss to get fixed
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', \
                         input_shape = (8, 8, 2))) # kernel_regularizer = l2(0.01)
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        # Add more Convolutional layers and Residual Blocks as needed

        # Flatten and fully connected layers
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation = 'relu'))
        # Value head

        model.add(Dense(1, activation = 'sigmoid'))

        # # Compile the model with appropriate loss functions and optimizer
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # Small Model
        # model = Sequential([
        #     Conv2D(32, (3, 3), activation = 'relu', input_shape = (8, 8, 2), padding = 'same'),
        #     Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
        #     Dropout(0.45),
        #     Flatten(),
        #     Dense(128, activation='relu'),
        #     Dense(1, activation='linear')
        # ])
        # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        return model

def generate_new_data(data_set_length, move_depth = 0, model = None):
    black_start = (1 << 28) | (1 << 35)
    white_start = (1 << 27) | (1 << 36)

    current_game_positions = []
    current_game_player_history = []

    positions = []
    final_scores = []  # Array of game scores corresponding to the positions

    while len(evaluations) < data_set_length:
        game_state = OthelloGameState(black_start, white_start, 1)
        
        # Get a rather random starting position
        for _ in range(6):
            random_move = choose_random_move(game_state)
            game_state = handle_legal_move(game_state, random_move)

            random_number = random.randint(1, 3)
            if random_number == 1:
                new_positions = get_position_arrays(game_state)
                current_game_positions.append(new_positions)
                current_game_player_history.append(game_state.current_player)
        
        two_turns_passed = False
        while not two_turns_passed:
            random_integer = random.randint(1, 10)
            if random_integer == 1:
                # This random move guaruntees variation in the data ontop of starting position
                # but also mean won't give away corner since is depth 2
                move = iterative_deepening_search(game_state, 2, False)
            elif move_depth == -1:
                move, _ = perfect_end_search(game_state, model)
            else:
                depth = move_depth
                if (64 - number_of_bits_set(game_state.black_bitboard | game_state.white_bitboard)) < 10:
                    depth = 9 # Get perfect play for last few moves
                move, _ = iterative_deepening_search(game_state, depth, False)

            game_state = handle_legal_move(game_state, move)

            if not find_legal_moves(game_state):
                opposite_player = 2 if game_state.current_player == 1 else 1
                game_state.current_player = opposite_player # Pass the turn
                if not find_legal_moves(game_state):
                    two_turns_passed = True
                    break
            # Pass turn before adding data
            random_number = random.randint(1, 3)
            if random_number == 1:
                new_positions = get_position_arrays(game_state)
                current_game_positions.append(new_positions)
                current_game_player_history.append(game_state.current_player)

        no_black_discs = number_of_bits_set(game_state.black_bitboard)
        no_white_discs = number_of_bits_set(game_state.white_bitboard)
        final_game_score = no_black_discs - no_white_discs
        for current_player in current_game_player_history:
            # Evaluation from perspective of current_player
            evaluation = final_game_score if current_player == 1 else -final_game_score
            for _ in range(8):
                final_scores.append(evaluation)
        positions += current_game_positions
        
    positions = np.array(positions)
    final_scores = np.array(final_scores)
    data = {'positions': positions, 'final_scores': final_scores}
    return data

def get_position_arrays(game_state):
    # Will turn game state into array of shape (8, 8, 2) and then get the 8 orientations of the board
    # to get list of 8 of these arrays
    array_of_orientations = []

    player_bitboard = game_state.black_bitboard
    opponent_bitboard = game_state.white_bitboard
    if game_state.current_player == 2:
        player_bitboard = game_state.white_bitboard
        opponent_bitboard = game_state.black_bitboard

    for i in range(2):
        if i == 2:
            new_player_bitboard = reflect_bitboard(player_bitboard)
            new_opponent_bitboard = reflect_bitboard(opponent_bitboard)
        for _ in range(4):
            player_binary_string = format(new_player_bitboard, '064b')
            opponent_binary_string = format(new_opponent_bitboard, '064b')
            player_array = np.array([int(bit) for bit in player_binary_string], dtype = np.int8)
            opponent_array = np.array([int(bit) for bit in opponent_binary_string], dtype = np.int8)
            # Arrays given shape (8, 8, 1)
            player_array = player_array.reshape(8, 8, 1)
            opponent_array = opponent_array.reshape(8, 8, 1)
            position = np.dstack((player_array, opponent_array)) # Gets shape (8, 8, 2)
            array_of_orientations.append(position)
            new_player_bitboard = rotate_bitboard(new_player_bitboard) # Rotate 90 degrees
            new_opponent_bitboard = rotate_bitboard(new_opponent_bitboard)
    
    return array_of_orientations
        
def train_model(model, model_path, training_data, validation_data):
    training_positions = training_data['positions']
    training_scores = training_data['final_scores']
    validation_positions = validation_data['positions']
    validation_scores = validation_data['final_scores']
    
    # Use this callback to keep fitting until no more improvement
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2, restore_best_weights = True)
    
    # No need to update learning rate since are using adam optimiser

    model.fit(training_positions, training_scores, \
        validation_data = (validation_positions, validation_scores), \
        epochs = 20, batch_size = 64, callbacks = [early_stopping])

    model.save(model_path)

    return model

if __name__ == '__main__':
    MODEL_PATH = './backend/Othello/Neural_Net_Training/Trained_Models/depth7_large_model.keras'
    DATA_PATH = './backend/Othello/Neural_Net_Training/Training_Data/depth7_large_model.keras'
    MOVE_DEPTH = 4 # Always use heuristic evaluation to generate data. SET TO -1 TO USE NETWORK TO PLAY
    MODE = 'generate data' # Other modes are train and self update

    if MODE == 'generate data':
        data_length = 1000
        while True:
            new_data = generate_new_data(data_length, MOVE_DEPTH, MODEL_PATH)


    if TRAIN_FROM_SAVED_DATA:
        model = get_model(MODEL_PATH)
        data_path = './backend/Othello/Neural_Net_Training/Training_Data/depth_7_evaluations.npy'
        training_data = np.load(data_path, allow_pickle = True)
        positions = training_data.item()['positions']
        evaluations = training_data.item()['evaluations']
        scaler = preprocessing.MinMaxScaler()
        normalised_evaluations = scaler.fit_transform(evaluations.reshape(-1, 1))
        X_train, X_val, y_train, y_val = model_selection.train_test_split\
            (positions, normalised_evaluations, test_size = 0.2, random_state = 42)
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2, restore_best_weights = True)
        model.fit(X_train, y_train, validation_data = (X_val, y_val), \
                  epochs = 100, batch_size = 64, callbacks = [early_stopping])
        model.save(MODEL_PATH)
        exit()
    
    iteration_no = 0
    current_model = get_model(MODEL_PATH)
    training_data_history = [] # Will start to reuse old data at some point
    validation_data_history = []
    while True:
        iteration_no += 1
        print('New iteration. Iteration no. ',  iteration_no) # New data when new iteration
        
        if iteration_no % 8 == 0: # Goes through all the past data to retrain
            positions = np.vstack([data_entry['positions'] for data_entry in training_data_history]) # Stacks all along first axis
            evaluations = np.concatenate([data_entry['evaluations'] for data_entry in training_data_history], axis = 0)
            training_data = {'positions': positions, 'evaluations': evaluations}
            positions = np.vstack([data_entry['positions'] for data_entry in validation_data_history])
            evaluations = np.concatenate([data_entry['evaluations'] for data_entry in validation_data_history], axis = 0)
            validation_data = {'positions': positions, 'evaluations': evaluations}
        else:
            # Generate training and validation data separately so that they are not from the same games
            # This will help prevent overfitting when we apply early stopping
            training_data = generate_new_data(int(0.8*DATA_SET_LENGTH), EVALUATION_DEPTH, MOVE_DEPTH, current_model)
            validation_data = generate_new_data(int(0.2*DATA_SET_LENGTH), EVALUATION_DEPTH, MOVE_DEPTH, current_model)
            data_save_path = './backend/Othello/Neural_Net_Training/Training_Data/depth_7_evaluations.npy'
            save_data(training_data['positions'], training_data['evaluations'], data_save_path)
            save_data(validation_data['positions'], validation_data['evaluations'], data_save_path)
            print('New data save', iteration_no)
            training_data_history.append(training_data)
            validation_data_history.append(validation_data)

        current_model = train_model(current_model, MODEL_PATH, training_data, validation_data)
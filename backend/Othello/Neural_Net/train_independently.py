# This file generates new data and trains the model using that new data

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

def generate_new_data(data_set_length, evaluation_depth = 0, move_depth = 0, model = None):
    black_start = (1 << 28) | (1 << 35)
    white_start = (1 << 27) | (1 << 36)

    positions = []
    evaluations = []

    while len(evaluations) < data_set_length:
        game_state = OthelloGameState(black_start, white_start, 1)
        
        # Get a rather random starting position for any MOVE_DEPTH
        for _ in range(6):
            random_move = choose_random_move(game_state)
            game_state = handle_legal_move(game_state, random_move)
            # Will not have data for starting position since move symmetric

            random_number = random.randint(1, 3)
            if random_number == 1:
                # This means we won't be getting lots of positions from the same game
                # which is essential for overfitting (I've tested it)
                # Note that by combining with a random oritentation we further solve the issue
                positions, evaluations = \
                    add_to_data(game_state, positions, evaluations, evaluation_depth, model)
        
        two_turns_passed = False
        while not two_turns_passed:
            random_integer = random.randint(1, 10)
            if random_integer == 1:
                # This random move guaruntees variation in the data ontop of starting position
                move = choose_random_move(game_state)
            elif move_depth == 0:
                move = choose_random_move(game_state)
            else:
                move, _ = iterative_deepening_search(game_state, move_depth, False)

            game_state = handle_legal_move(game_state, move)

            if not find_legal_moves(game_state):
                opposite_player = 2 if game_state.current_player == 1 else 1
                game_state.current_player = opposite_player # Pass the turn
                if not find_legal_moves(game_state):
                    two_turns_passed = True
            # Pass turn before adding data (doesn't actually affect data quality)
            random_number = random.randint(1, 3)
            if random_number == 1:
                positions, evaluations = \
                    add_to_data(game_state, positions, evaluations, evaluation_depth, model)
        
    positions = np.array(positions)
    evaluations = np.array(evaluations)
    data = {'positions': positions, 'evaluations': evaluations}
    return data

def add_to_data(game_state, position_array, evaluation_array, evaluation_depth, model = None):
    if len(evaluation_array) % 200 == 0:
        print(len(position_array), 'data entries')
    if evaluation_depth == -1: # This means using neural network to evaluate the position 
        _, evaluation = perfect_end_search(game_state, model)
        evaluation = evaluation * 64 - 32 # Scale back up for storing in desired format
    elif evaluation_depth == 0:
        current_player = game_state.current_player
        evaluation = heuristic_evaluation(game_state)
        game_state.current_player = current_player 
        # ^ Avoid issue with heuristic_eval changing current_player
    else: # i.e. evaluation depth > 0
        _, evaluation = iterative_deepening_search(game_state, evaluation_depth, False)
    
    player_bitboard = game_state.black_bitboard
    opponent_bitboard = game_state.white_bitboard
    if game_state.current_player == 2:
        player_bitboard = game_state.white_bitboard
        opponent_bitboard = game_state.black_bitboard
        evaluation = -evaluation

    # heuristic_eval can give eval up to 70 but we restrict to this range for
    # when normalising the data
    if evaluation > 30:
        evaluation = min(32, 30 + evaluation / 100)
    if evaluation < -30:
        evaluation = max(-32, -30 + evaluation / 100)

    random_number = random.randint(1, 4)
    new_player_bitboard = player_bitboard
    new_opponent_bitboard = opponent_bitboard
    for i in range(2):
        if i == 2:
            new_player_bitboard = reflect_bitboard(player_bitboard)
            new_opponent_bitboard = reflect_bitboard(opponent_bitboard)
        for _ in range(1, 1 + random_number % 4):
            new_player_bitboard = rotate_bitboard(new_player_bitboard)
            new_opponent_bitboard = rotate_bitboard(new_opponent_bitboard)
        player_binary_string = format(new_player_bitboard, '064b')
        opponent_binary_string = format(new_opponent_bitboard, '064b')
        player_array = np.array([int(bit) for bit in player_binary_string], dtype = np.int8)
        opponent_array = np.array([int(bit) for bit in opponent_binary_string], dtype = np.int8)
        # Arrays given shape (8, 8, 1)
        player_array = player_array.reshape(8, 8, 1)
        opponent_array = opponent_array.reshape(8, 8, 1)
        position = np.dstack((player_array, opponent_array)) # Gets shape (8, 8, 2)
        position_array.append(position)
        evaluation_array.append(evaluation)
    return position_array, evaluation_array
        
def train_model(model, model_path, training_data, validation_data):
    scaler = preprocessing.MinMaxScaler()
    training_positions = training_data['positions']
    training_evaluations = training_data['evaluations']
    validation_positions = validation_data['positions']
    validation_evaluations = validation_data['evaluations']

    normalised_training_evaluations = scaler.fit_transform(training_evaluations.reshape(-1, 1))
    normalised_validation_evaluations = scaler.fit_transform(validation_evaluations.reshape(-1, 1))
    
    # Use this callback to keep fitting until no more improvement
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 4, restore_best_weights = True)
    
    # No need to update learning rate since are using adam optimiser

    model.fit(training_positions, normalised_training_evaluations, \
        validation_data = (validation_positions, normalised_validation_evaluations), \
        epochs = 100, batch_size = 64, callbacks = [early_stopping])

    model.save(model_path)

    return model

if __name__ == '__main__':
    MODEL_PATH = './backend/Othello/Neural_Net_Training/Trained_Models/depth7_large_model.keras'
    DATA_SET_LENGTH = 20000
    EVALUATION_DEPTH = 7 # Set to -1 if want to use neural network to evaluate positions
    MOVE_DEPTH = 3 # Always use heuristic evaluation to generate data
    TRAIN_FROM_SAVED_DATA = True

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
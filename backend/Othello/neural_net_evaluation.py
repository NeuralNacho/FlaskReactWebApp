'''We implement a different search algorithm to make use of batch processing
Based on the speed of batch processing, it appears that a depth 4 minimax search,
which is normally < 10^4 = 10,000 nodes, is most appropriate for the final function
since takes 0.6s for 9600 evaluations and is slower (per evaluation) for smaller
numbers of evaluations while minimax on these 10,000 nodes is < 0.2s so pretty small.
If the speed of a single evaluation (currently 0.085s per eval) were say 100x faster
then MCTS would be an interesting approach '''

import numpy as np
from .game_info_functions import *
from .heuristic_evaluation import *

def perfect_end_search(game_state, model):
    if 64 - number_of_bits_set(game_state.black_bitboard | game_state.white_bitboard) < 10:
        # Get perfect move for end of game when less options
        best_move, evaluation, number_of_nodes = iterative_deepening_search(game_state, 9)

        # Normalise the data
        if evaluation > 30:
            evaluation = min(32, 30 + evaluation / 100)
        if evaluation < -30:
            evaluation = max(-32, -30 + evaluation / 100)
        evaluation += 32
        evaluation /= 64
        return best_move, evaluation, number_of_nodes
    else:
        root_node, evaluation, number_of_nodes = batch_minimax_search(game_state, model)

        # Get the best move
        for child_node in root_node.children:
            if child_node.evaluation == evaluation: # i.e. This child is the best move
                root_node_bitboard = root_node.game_state.black_bitboard | root_node.game_state.white_bitboard
                child_node_bitboard = child_node.game_state.black_bitboard | child_node.game_state.white_bitboard
                best_move = root_node_bitboard ^ child_node_bitboard
                break

    return best_move, evaluation, number_of_nodes

def evaluation_search(game_state, model): # For getting eval of all legal moves
    if 64 - number_of_bits_set(game_state.black_bitboard | game_state.white_bitboard) < 10:
        # Get perfect move for end of game when less options
        evaluations_board, number_of_nodes = iterative_deepening_evaluation_search(game_state, 9)
        return evaluations_board, number_of_nodes
    else:
        root_node, evaluation, number_of_nodes = batch_minimax_search(game_state, model)
        root_node_bitboard = root_node.game_state.black_bitboard | root_node.game_state.white_bitboard
        evaluations_board = [['' for _ in range(8)] for _ in range(8)]
        for child_node in root_node.children:
            child_node_bitboard = child_node.game_state.black_bitboard | child_node.game_state.white_bitboard
            move = child_node_bitboard ^ root_node_bitboard
            move_index = get_move_index(move)
            row = move_index[0]
            col = move_index[1]
            evaluation = child_node.evaluation*64 - 32
            evaluation = int(evaluation)
            evaluations_board[row][col] = str(evaluation)
    
    return evaluations_board, number_of_nodes

def batch_minimax_search(game_state, model):
    # 'batch' since all leaf nodes evaluated at once in a batch
    if number_of_bits_set(find_legal_moves(game_state)) >= 10:
        # If search tree is too large depth 4 will take long time to predict
        search_depth = 3
    else:
        search_depth = 4
    root_node = MinimaxTreeNode(game_state, search_depth)
    leaf_nodes = []
    explore_tree(root_node, leaf_nodes)

    # Prepare leaf nodes for batch evaluation
    leaf_node_positions_array = []
    for node in leaf_nodes:
        if node.game_state.current_player == 1:
            player_bitboard = node.game_state.black_bitboard
            opponent_bitboard = node.game_state.white_bitboard
        if node.game_state.current_player == 2:
            player_bitboard = node.game_state.white_bitboard
            opponent_bitboard = node.game_state.black_bitboard
        player_binary_string = format(player_bitboard, '064b')
        opponent_binary_string = format(opponent_bitboard, '064b')
        player_array = np.array([int(bit) for bit in player_binary_string], dtype = np.int8)
        opponent_array = np.array([int(bit) for bit in opponent_binary_string], dtype = np.int8)
        # Arrays given shape (8, 8, 1)
        player_array = player_array.reshape(8, 8, 1)
        opponent_array = opponent_array.reshape(8, 8, 1)
        position = np.dstack((player_array, opponent_array)) # Gets shape (8, 8, 2)
        leaf_node_positions_array.append(position)

    # Evaluate the leaf nodes
    leaf_node_positions_array = np.array(leaf_node_positions_array)
    leaf_node_evaluations = model.predict(leaf_node_positions_array)

    for index, node in enumerate(leaf_nodes):
        if node.game_state.current_player == 1:
            node.evaluation = leaf_node_evaluations[index][0]
            # [0] because for some reason .predict encloses each entry in a []
        else: # In this case the current player is white and we need to invert eval
            # since our prediction is based on how good the position is for current player
            node.evaluation = 1 - leaf_node_evaluations[index][0]
    
    # Propagate evaluation up the tree
    evaluation = get_node_evaluation(root_node)
    
    number_of_nodes = len(leaf_nodes) 
    number_of_nodes += (number_of_nodes / 8) # Approximate nodes in second last layer

    return root_node, evaluation, number_of_nodes

class MinimaxTreeNode:
    def __init__(self, game_state, depth):
        self.game_state = game_state
        self.children = []
        self.evaluation = -1
        self.depth = depth # 0 if leaf node

def explore_tree(root_node, leaf_nodes):
    # Will find all the leaf nodes for a minimax search
    if root_node.depth == 0:
        if not find_legal_moves(root_node.game_state):
            # Be nice and switch the player for the eval function
            # (Though a good eval would see no moves for current player anyway)
            root_node.game_state.current_player = 1 if root_node.game_state.current_player == 2 else 2
        leaf_nodes.append(root_node)
        return
    
    legal_moves = find_legal_moves(root_node.game_state)
    if not legal_moves: # Deal with passing turns here
        root_node.game_state.current_player = 1 if root_node.game_state.current_player == 2 else 2
        legal_moves = find_legal_moves(root_node.game_state)
        if not legal_moves: # In this case, game is over
            leaf_nodes.append(root_node)
            return
    
    while legal_moves:
        move = legal_moves & -legal_moves
        legal_moves ^= move
        new_game_state = OthelloGameState(root_node.game_state.black_bitboard, \
            root_node.game_state.white_bitboard, root_node.game_state.current_player)
        new_game_state = handle_legal_move(new_game_state, move)
        child_node = MinimaxTreeNode(new_game_state, root_node.depth - 1)
        root_node.children.append(child_node)
        explore_tree(child_node, leaf_nodes)

def get_node_evaluation(node):
    # Will go back from the leaf nodes to the origin node
    # Could use alpha beta search alg but this is not a bottleneck so keep it simpler with minimax
    if len(node.children) == 0:
        return node.evaluation
    
    if node.game_state.current_player == 1:
        best_value = float('-inf')
        for child in node.children:
            value = get_node_evaluation(child)
            best_value = max(best_value, value)
        node.evaluation = best_value 
        # ^ Important to populate evaluations of the nodes for selecting best move from root node
        return best_value
    else:
        best_value = float('inf')
        for child in node.children:
            value = get_node_evaluation(child)
            best_value = min(best_value, value)
        node.evaluation = best_value
        return best_value
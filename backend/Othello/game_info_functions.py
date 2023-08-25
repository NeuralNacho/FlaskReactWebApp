class OthelloGameState:
    def __init__(self, black_bitboard, white_bitboard, current_player):
        # Three game state variables:
        self.black_bitboard = black_bitboard
        self.white_bitboard = white_bitboard
        self.current_player = current_player

class OthelloSearchState:
    # Class used to store information about the search for each node
    # in the alpha beta search
    def __init__(self, best_moves, hash_moves, depth, max_depth, alpha = float('-inf'), \
                 beta = float('inf'), node_count = 0):
        self.best_moves = best_moves
        self.hash_moves = hash_moves
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.max_depth = max_depth
        self.node_count = node_count
        # Node count is how many nodes we've checked before this one.
        # Will be used to display the number of nodes checked

def get_game_state(board, current_player):
    black_bitboard, white_bitboard = get_bitboards(board)
    return OthelloGameState(black_bitboard, white_bitboard, current_player)

def get_bitboards(board):
    black_bitboard = 0
    white_bitboard = 0

    for row in range(8):
        for col in range(8):
            index = 63 - (row * 8 + col)
            # '63 -' since our bitboard is board array reversed 
            if board[row][col] == 1:
                black_bitboard |= 1 << index
            elif board[row][col] == 2:
                white_bitboard |= 1 << index
    return black_bitboard, white_bitboard

def find_legal_moves(game_state):
    not_left_side = 0x7F7F7F7F7F7F7F7F
    not_right_side = 0xFEFEFEFEFEFEFEFE
    legal_moves_bitboard = 0
    player_bitboard = game_state.black_bitboard if game_state.current_player == 1 \
        else game_state.white_bitboard
    opponent_bitboard = game_state.black_bitboard if game_state.current_player == 2 \
        else game_state.white_bitboard

    def up_left_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        # Get the empty spaces
        checker ^= player_bitboard | opponent_bitboard

        # Get all empty spaces with opponent disc up and left
        # Also make sure discs aren't wrapped around for either player's board
        opponent_bitboard_shifted = not_left_side & (opponent_bitboard >> 9)
        player_bitboard_shifted = not_left_side & (player_bitboard >> 9)
        checker &= opponent_bitboard_shifted
        while checker:
            # Get some new legal moves
            player_bitboard_shifted = not_left_side & (player_bitboard_shifted >> 9)
            legal_move_bitboard |= checker & (player_bitboard_shifted)
            # Get all empty spaces with a number of opponet discs up and left
            opponent_bitboard_shifted = not_left_side & (opponent_bitboard_shifted >> 9)
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def up_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = opponent_bitboard >> 8
        player_bitboard_shifted = player_bitboard >> 8
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted >>= 8
            legal_move_bitboard |= checker & player_bitboard_shifted
            opponent_bitboard_shifted >>= 8
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def up_right_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = not_right_side & (opponent_bitboard >> 7)
        player_bitboard_shifted = not_right_side & (player_bitboard >> 7)
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted = not_right_side & (player_bitboard_shifted >> 7)
            legal_move_bitboard |= checker & (player_bitboard_shifted)
            opponent_bitboard_shifted = not_right_side & (opponent_bitboard_shifted >> 7)
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def left_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = not_left_side & (opponent_bitboard >> 1)
        player_bitboard_shifted = not_left_side & (player_bitboard >> 1)
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted = not_left_side & (player_bitboard_shifted >> 1)
            legal_move_bitboard |= checker & (player_bitboard_shifted)
            opponent_bitboard_shifted = not_left_side & (opponent_bitboard_shifted >> 1)
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def right_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = not_right_side & (opponent_bitboard << 1)
        player_bitboard_shifted = not_right_side & (player_bitboard << 1)
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted = not_right_side & (player_bitboard_shifted << 1)
            legal_move_bitboard |= checker & (player_bitboard_shifted)
            opponent_bitboard_shifted = not_right_side & (opponent_bitboard_shifted << 1)
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def down_left_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = not_left_side & (opponent_bitboard << 7)
        player_bitboard_shifted = not_left_side & (player_bitboard << 7)
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted = not_left_side & (player_bitboard_shifted << 7)
            legal_move_bitboard |= checker & (player_bitboard_shifted)
            opponent_bitboard_shifted = not_left_side & (opponent_bitboard_shifted << 7)
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def down_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = opponent_bitboard << 8
        player_bitboard_shifted = player_bitboard << 8
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted <<= 8
            legal_move_bitboard |= checker & player_bitboard_shifted
            opponent_bitboard_shifted <<= 8
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    def down_right_finder():
        legal_move_bitboard = 0
        checker = (1 << 64) - 1
        checker ^= player_bitboard | opponent_bitboard
        opponent_bitboard_shifted = not_right_side & (opponent_bitboard << 9)
        player_bitboard_shifted = not_right_side & (player_bitboard << 9)
        checker &= opponent_bitboard_shifted
        while checker:
            player_bitboard_shifted = not_right_side & (player_bitboard_shifted << 9)
            legal_move_bitboard |= checker & (player_bitboard_shifted)
            opponent_bitboard_shifted = not_right_side & (opponent_bitboard_shifted << 9)
            checker &= (opponent_bitboard_shifted)
        return legal_move_bitboard

    legal_moves_bitboard |= up_left_finder()
    legal_moves_bitboard |= up_finder()
    legal_moves_bitboard |= up_right_finder()
    legal_moves_bitboard |= left_finder()
    legal_moves_bitboard |= right_finder()
    legal_moves_bitboard |= down_left_finder()
    legal_moves_bitboard |= down_finder()
    legal_moves_bitboard |= down_right_finder()
    return legal_moves_bitboard

def handle_legal_move(game_state, move_bitboard):
    # Add the disc placed to the correct colour's board
    if game_state.current_player == 1:
        game_state.black_bitboard |= move_bitboard
    else:
        game_state.white_bitboard |= move_bitboard

    game_state = flip_discs(game_state, move_bitboard)
    game_state.current_player = 1 if game_state.current_player == 2 else 2
    # DON'T deal with passing of turn here
    return game_state

def flip_discs(game_state, move_bitboard):
    # Magic bitboards could be used to speed this up big time but not worth for now
    not_left_side = 0x7F7F7F7F7F7F7F7F
    not_right_side = 0xFEFEFEFEFEFEFEFE
    player_bitboard = game_state.black_bitboard if game_state.current_player == 1 \
        else game_state.white_bitboard
    opponent_bitboard = game_state.black_bitboard if game_state.current_player == 2 \
        else game_state.white_bitboard
    
    def flip_up_left(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard << 9
        while opponent_bitboard & index & not_right_side:
            discs_to_flip |= index
            index <<= 9
        # Place on player bitboard and delete disc on opponent bitboard
        if player_bitboard & index & not_right_side:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard

    def flip_up(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard << 8
        while opponent_bitboard & index:
            discs_to_flip |= index
            index <<= 8
        if player_bitboard & index:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard
    
    def flip_up_right(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard << 7
        while opponent_bitboard & index & not_left_side:
            discs_to_flip |= index
            index <<= 7
        if player_bitboard & index & not_left_side:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard
    
    def flip_left(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard << 1
        while opponent_bitboard & index & not_right_side:
            discs_to_flip |= index
            index <<= 1
        if player_bitboard & index & not_right_side:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard

    def flip_right(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard >> 1
        while opponent_bitboard & index & not_left_side:
            discs_to_flip |= index
            index >>= 1
        if player_bitboard & index & not_left_side:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard

    def flip_down_left(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard >> 7
        while opponent_bitboard & index & not_right_side:
            discs_to_flip |= index
            index >>= 7
        if player_bitboard & index & not_right_side:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard

    def flip_down(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard >> 8
        while opponent_bitboard & index:
            discs_to_flip |= index
            index >>= 8
        if player_bitboard & index:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard

    def flip_down_right(player_bitboard, opponent_bitboard):
        discs_to_flip = 0
        index = move_bitboard >> 9
        while opponent_bitboard & index & not_left_side:
            discs_to_flip |= index
            index >>= 9
        if player_bitboard & index & not_left_side:
            player_bitboard |= discs_to_flip
            opponent_bitboard &= ~discs_to_flip
        return player_bitboard, opponent_bitboard
    
    player_bitboard, opponent_bitboard = flip_up_left(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_up(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_up_right(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_left(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_right(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_down_left(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_down(player_bitboard, opponent_bitboard)
    player_bitboard, opponent_bitboard = flip_down_right(player_bitboard, opponent_bitboard)

    game_state.black_bitboard = player_bitboard if game_state.current_player == 1\
        else opponent_bitboard
    game_state.white_bitboard = player_bitboard if game_state.current_player == 2\
        else opponent_bitboard
    return game_state

def get_move_index(move):
    # Function to go from move bitboard to row col index
    move = 64 - move.bit_length() # Gets move bit length
    # '64 -' since bitboard is reverse of board array
    col = move % 8
    row = move // 8
    move = [row, col]
    return move

def number_of_bits_set(bitboard):
    # Don't need to worry about having to copy bitboard since it is immuatble
    no_bits = 0
    while bitboard:
        no_bits += 1
        bitboard &= (bitboard - 1)
        # The minus 1 above will turn rightmost 1 bit to a 0 and the 
        # zeros to the right of that into ones
    return no_bits

def print_bitboard(bitboard):
    # Useful for debugging
    # Note that here you consider the leftmost bit the top left corner
    for row in range(8):
        row_string = ""
        for col in range(8):
            if bitboard & (1 << (63 - (row*8 + col))):
                row_string += "X "
            else:
                row_string += ". "
        print(row_string)

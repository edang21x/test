import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import Counter

import numpy as np

from game_mechanics import (
    Connect4Env,
    choose_move_randomly,
    has_won,
    is_column_full,
    load_dictionary,
    place_piece,
    play_connect_4_game,
    reward_function,
    human_player,
)

TEAM_NAME = "Henry"  # <---- Enter your team name here!


def get_poss_fours_num_filled(board: np.ndarray, player: int) -> List[int]:
    my_piece_locations = [(row, col) for row in range(board.shape[0]) for col in range(board.shape[1]) if
                          board[row, col] == player]
    # NOTE: Not checking for vertical possible 4's! Only diagonals and horizontal
    #  Verticals are the worst
    directions = [
        [0, 1],
        [-1, 1],
        [-1, -1],
    ]

    fours_added = set()
    four_num_filled = []
    for piece_location in my_piece_locations:
        # Try all directions
        for direction in directions:
            # We're looking for a possible 4 through this piece,
            #  start at the piece with a line of 1
            num_pieces_filled = 1
            line_piece_locations = [(piece_location[0], piece_location[1])]

            # Try spaces in positive direction
            steps_in_positive_dir = 1

            # Take steps in positive direction until we hit a space not filled by this player's piece
            row = piece_location[0] + steps_in_positive_dir * direction[0]
            col = piece_location[1] + steps_in_positive_dir * direction[1]
            while (
                    0 <= row < board.shape[0]
                    and 0 <= col < board.shape[1]
                    and board[row, col] in [player, 0]
                    and len(line_piece_locations) < 4
            ):
                num_pieces_filled += 1 if board[row, col] == player else 0
                line_piece_locations.append((row, col))
                steps_in_positive_dir += 1
                row = piece_location[0] + steps_in_positive_dir * direction[0]
                col = piece_location[1] + steps_in_positive_dir * direction[1]

            # Try spaces in negative direction
            steps_in_negative_dir = 1

            row = piece_location[0] - steps_in_negative_dir * direction[0]
            col = piece_location[1] - steps_in_negative_dir * direction[1]
            while (
                    row in range(board.shape[0])
                    and col in range(board.shape[1])
                    and board[row, col] in [player, 0]
                    and len(line_piece_locations) < 4
            ):
                num_pieces_filled += 1 if board[row, col] == player else 0
                line_piece_locations.append((row, col))
                steps_in_negative_dir += 1
                row = piece_location[0] - steps_in_negative_dir * direction[0]
                col = piece_location[1] - steps_in_negative_dir * direction[1]

            # Add all row lengths in each direction to the list
            line_piece_locations = tuple(sorted(line_piece_locations, key=lambda x: 10 * x[0] + x[1]))
            if len(line_piece_locations) >= 4 and line_piece_locations not in fours_added:
                fours_added.add(line_piece_locations)
                four_num_filled.append(num_pieces_filled)

    return four_num_filled


def to_feature_vector(state: np.ndarray, prev_action: Optional[int], player_turn: int = -1) -> Tuple:
    # Terminal state check
    if prev_action is not None and (has_won(state, prev_action) or np.count_nonzero(state) == 48):
        return "Terminal State",

    # First 2 moves - show middle 4 columns of the bottom 2 rows
    if np.count_nonzero(state) <= 2 * 2:
        return tuple(state[4:, 2:6].flatten())

    # Get possible actions
    not_full_cols = [col for col in range(state.shape[1]) if not is_column_full(state, col)]

    # Check all possible moves to see if opponent wins anywhere
    loses = []
    for col_idx in not_full_cols:
        state_copy = state.copy()
        place_piece(state_copy, col_idx, -1)
        if has_won(state_copy, col_idx):
            loses.append(col_idx)

    if len(loses) > 1:
        return 2,  # More than 2 losing positions is equivalent to 2
    elif len(loses) and player_turn == -1:
        return len(loses), player_turn

    # Find win spots & check if space above is also a win
    win_move_count = 0
    for col_idx in not_full_cols:
        # Copy so it doesn't affect the actual state variable
        state_copy = state.copy()
        state_copy, row_idx = place_piece(state_copy, col_idx)
        if has_won(state_copy, col_idx):
            win_move_count += 1

            # Check for 2 win spots above one another if the win isn't not a vertical 3
            if not is_column_full(state_copy, col_idx) and (row_idx > 2 or not np.alltrue(state_copy[row_idx : row_idx + 4, col_idx])):
                place_piece(state_copy, col_idx)
                if has_won(state_copy, col_idx):
                    win_move_count += 1

    # Get the number of possible 4's for us
    possible_four_piece_counts = Counter(get_poss_fours_num_filled(state, 1))
    num_threes = possible_four_piece_counts.get(3, 0)
    num_twos = possible_four_piece_counts.get(2, 0)

    # Get the number of possible 4's for them
    oppo_poss_four_counts = Counter(get_poss_fours_num_filled(state, -1))
    num_oppo_threes = oppo_poss_four_counts.get(3, 0)

    # Include player_turn, since we're updating states associated with both players' turns.
    #  This is beneficial since you see the states your opponent sees in 1-step lookahead.
    return num_threes, num_twos, num_oppo_threes, win_move_count, len(loses), player_turn


def train(value_fn: Optional[Dict] = None, verbose: bool = False) -> Dict:
    epsilon = 0.25
    alpha = 0.25
    epsilon_decay = 0.999
    alpha_decay = 0.9999
    value_fn = {} if value_fn is None else value_fn

    games_won, games_lost = 0, 0
    for _ in tqdm(range(10000)):
        game = Connect4Env(lambda x: epsilon_greedy_policy(x, epsilon, value_fn))
        state, reward, done, info = game.reset()
        features = to_feature_vector(state, None, 1)
        while not done:
            if verbose:
                print("\n", state, "\n", features)
            action = epsilon_greedy_policy(state, epsilon, value_fn, verbose)

            prev_feat = features
            state_copy = state.copy()
            place_piece(board=state_copy, column_idx=action, player=1)
            state, reward, done, _ = game.step(action)

            if np.count_nonzero(state_copy - state) == 1:
                last_move = (np.flatnonzero(state_copy - state) % 8).item()
            else:
                last_move = action

            # Update the state you started the step in
            features = to_feature_vector(state, last_move, 1)
            value_fn[prev_feat] = (1 - alpha) * value_fn.get(prev_feat, 0) + alpha * (reward + value_fn.get(features, 0))

            # Unless you just played the winning move (reward == 1) when only 1 move is taken
            if reward != 1:
                # Update the post-move state value. These will be used in 1-step lookahead.
                post_move_feat = to_feature_vector(state_copy, action, -1)
                value_fn[post_move_feat] = (1 - alpha) * value_fn.get(post_move_feat, 0) + alpha * (reward + value_fn.get(features, 0))

        # Decay alpha & epsilon
        alpha = max(alpha * alpha_decay, 0.1)
        epsilon = max(epsilon * epsilon_decay, 0.01)

        # Just vaguely tracking win/loss rate against itself during training.
        # Expecting ~50:50 since it's playing itself.
        games_won += 1 if reward == 1 else 0
        games_lost += 1 if reward == -1 else 0

    print(games_won, games_lost)

    return value_fn


def choose_move(state: np.ndarray, value_function: Dict, verbose: bool = False) -> int:
    """
    Called during competitive play. It acts greedily given
    current state of the board and value function dictionary.
    It returns a single move to play.

    Args:
        state: State of the board as a np array. Your pieces are
                1's, the opponent's are -1's and empty are 0's.
        value_function: The dictionary output by train().

    Returns:
        position (int): The column you want to place your counter
                        into (an integer 0 -> 7), where 0 is the
                        far left column and 7 is the far right
                        column.
    """
    values = []
    not_full_cols = [col for col in range(state.shape[1]) if not is_column_full(state, col)]

    for not_full_col in not_full_cols:
        # Do 1-step lookahead and compare values of successor states
        state_copy = state.copy()
        place_piece(board=state_copy, column_idx=not_full_col, player=1)

        # Get the feature vector associated with the successor state
        features = to_feature_vector(state_copy, not_full_col, -1)
        if verbose:
            print(
                "Col",
                not_full_col,
                "Feature:",
                features,
                "Value:",
                value_function.get(features, 0),
            )

        # Add the value of the sucessor state to the values list
        action_value = value_function.get(features, 0) + reward_function(not_full_col, state_copy)
        values.append(action_value)

    # Pick randomly between actions that have successor states with the maximum value
    max_value = max(values)
    value_indices = [index for index, value in enumerate(values) if value == max_value]
    value_index = random.choice(value_indices)
    return not_full_cols[value_index]


def epsilon_greedy_policy(state: np.ndarray, eps: float, value_fn: Dict, verbose: bool = False) -> int:
    if verbose:
        print("Selecting an action!")
    if random.random() < eps:
        if verbose:
            print("Epsilon hit! Random choice made!")
        return choose_move_randomly(state)
    else:
        return choose_move(state, value_fn, verbose)


if __name__ == "__main__":
    # Can either start from scratch or a previous attempt.
    # my_value_fn = {}
    # my_value_fn = load_dictionary(TEAM_NAME)

    # I edited train() to accept pre-trained value function
    # my_value_fn = train(my_value_fn, True)
    # save_dictionary(my_value_fn, TEAM_NAME)
    my_value_fn = load_dictionary(TEAM_NAME)
    # print(my_value_fn[("Terminal State", )])
    print("len(my_value_fn)", len(my_value_fn))

    # Checking the "they're about to win" feature is negative and reasonably close to -1
    print("my_value_fn[(1, -1)]", my_value_fn[(1, -1)], my_value_fn[(2, )])

    n_wins, n_draws, n_losses = 0, 0, 0
    for _ in tqdm(range(1000)):
        reward = play_connect_4_game(
            your_choose_move=lambda x: choose_move(x, my_value_fn),
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=1000000000,
            render=False,
            verbose=False,
        )
        n_wins += 1 if reward == 1 else 0
        n_draws += 1 if reward == 0 else 0
        n_losses += 1 if reward == -1 else 0
    print("Num wins:", n_wins, "Num draws:", n_draws, "Num losses:", n_losses)

    play_connect_4_game(
        your_choose_move=lambda x: choose_move(x, my_value_fn, True),
        opponent_choose_move=human_player,
        game_speed_multiplier=1000,
        render=True,
        verbose=True,
    )
from typing import Dict, Tuple
from tqdm import tqdm
from collections import deque
import random

# import matplotlib.pyplot as plt
import statistics

import numpy as np
from torch import nn
import torch

from check_submission import check_submission
from game_mechanics import (
    OthelloEnv,
    choose_move_randomly,
    human_player,
    load_network,
    play_othello_game,
    save_network,
    get_legal_moves,
    make_move,
    reward_function,
)

TEAM_NAME = "Anudeep"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

# Hyperparameters
epsilon = 0.3
gamma = 0.95
lr = 1e-2
batch_size = 32
N = 1000

num_of_episodes = 1000  # 50 # change later to a higher value
layer_1_size = 20
layer_2_size = 12


def flatten_board(board: np.ndarray) -> np.ndarray:
    return board.reshape(36)


def choose_move_greedily(
    value_fn: nn.Module,
    board: np.ndarray,
    gamma: float = gamma,
):
    """choose greedy move that maximizes value"""
    poss_moves = get_legal_moves(board)
    # print(len(poss_moves), " --> ", poss_moves)

    if len(poss_moves) == 0:
        return None

    successor_boards = [make_move(board, move) for move in poss_moves]
    rewards = [reward_function(board) for board in successor_boards]

    # flatten board
    flattened_successor_boards = [flatten_board(board) for board in successor_boards]

    # convert into tensor
    rewards = torch.tensor(rewards, dtype=torch.float32)
    flattened_successor_boards = torch.tensor(
        np.array(flattened_successor_boards), dtype=torch.float32
    )

    # print("value_fn(flattened_successor_boards)",value_fn(flattened_successor_boards).shape)
    values = torch.squeeze(value_fn(flattened_successor_boards))

    action_values = rewards + gamma * values

    chosen_action = poss_moves[torch.argmax(action_values)]
    return chosen_action


def train() -> nn.Module:
    """
    TODO: Write this function to train your network.
    """
    # plot the loss
    loss_list = []
    num_written_to_memory = 0
    num_written_per_loop = []

    # initialize the neural network
    V = nn.Sequential(
        nn.Linear(36, layer_1_size),
        nn.ReLU(),
        nn.Linear(layer_1_size, layer_2_size),
        nn.ReLU(),
        nn.Linear(layer_2_size, 1),
    )

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(V.parameters(), lr=lr)

    # set up the env and the memory store
    env = OthelloEnv()
    memory_store = deque(maxlen=N)

    for _ in tqdm(range(num_of_episodes)):

        board, reward, done, _ = env.reset()

        while not done:
            prev_board = board
            move = (
                choose_move_randomly(prev_board)
                if random.random() < epsilon
                else choose_move_greedily(V, prev_board, gamma)
            )
            board, reward, done, _ = env.step(move)

            memory_store.append((flatten_board(prev_board), reward, flatten_board(board)))

            if batch_size < len(memory_store):
                #         num_written_per_loop.append(num_written_to_memory)
                batch = random.sample(memory_store, batch_size)
                x = torch.stack([torch.tensor(board, dtype=torch.float32) for board, _, _ in batch])
                #         print('--'*10, " ", "input value shape", x.shape, '--'*10, end = '\n')
                predicted_values = V(x)
                #         print('--'*10, " ", "predicted value shape", predicted_values.shape, '--'*10, end = '\n')

                # build target values
                reward_batch = torch.unsqueeze(
                    torch.tensor([reward for _, reward, _ in batch], dtype=torch.float32), dim=1
                )
                #         print("reward shape", reward_batch.shape)
                flat_successor_boards_batch = torch.stack(
                    [
                        torch.tensor(successor_board, dtype=torch.float32)
                        for _, _, successor_board in batch
                    ]
                )
                #         print("flat successor shape", flat_successor_boards_batch.shape)

                with torch.no_grad():
                    target = reward_batch + gamma * V(flat_successor_boards_batch)
                #         print("target shape", target.shape)

                loss = loss_fn(predicted_values, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                #         print("loss --> ", loss)

                loss_list.append(loss.item())
            num_written_to_memory += 1
        num_written_to_memory += 1
    #   print("num written per loop", len(num_written_per_loop), num_written_per_loop)
    #   print(loss_list)
    #    print(max(loss_list), min(loss_list), statistics.median(loss_list), statistics.stdev(loss_list))
    #    plt.plot(loss_list)
    return V


def choose_move(
    state: np.ndarray,
    neural_network: nn.Module,
) -> Tuple[int, int]:
    """Called during competitive play.

    It acts greedily given current state of the board and
    your neural_netwok. It returns a single move to play.

    Args:
        state: [6x6] np array defining state of the board
        neural_network: Your network from train()
    Returns:
        move: (row, col) position to place your piece
    """
    return choose_move_greedily(neural_network, state)


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    # neural_network = train()
    # save_network(neural_network, TEAM_NAME)

    # Ensure this passes in Replit!!
    check_submission(TEAM_NAME)

    my_network = load_network(TEAM_NAME)
    my_network2 = load_network("adversarial_mimic")

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(state: np.ndarray) -> Tuple[int, int]:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_network)

    def second_network_to_mimic_human(state: np.ndarray) -> Tuple[int, int]:
        """I use this second network to test out different strategies.
        Used it to mimic a human because my render was not working, but eventually used it to test out different network weights.
        """
        return choose_move(state, my_network2)

    play_othello_game(
        your_choose_move=second_network_to_mimic_human,  # choose_move_randomly,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        verbose=True,
        render=False,
    )
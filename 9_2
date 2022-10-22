from typing import Optional

from tqdm import tqdm
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from env import PendulumEnv, transition_function, reward_function
from utils import save_network, load_network, validate


def act_greedily(
    value_fn: nn.Module,
    state: torch.Tensor,
    gamma: float,
    writer: Optional = None,
    count: Optional[int] = None,
):
    """Act greedily, given a value function and state."""
    poss_actions = PendulumEnv.action_space
    successor_states = torch.stack(
        [transition_function(state, poss_action) for poss_action in poss_actions]
    )
    rewards = torch.tensor(
        [
            reward_function(poss_action, successor_state)
            for poss_action, successor_state in zip(poss_actions, successor_states)
        ]
    )

    values = torch.squeeze(value_fn(successor_states))

    action_values = gamma * values + rewards

    if writer is not None:
        writer.add_scalar("Values/-1", values[0], count)
        writer.add_scalar("Values/+1", values[1], count)

    return poss_actions[torch.argmax(action_values)]


gamma = 0.95


def train():
    # Hyperparams
    epsilon = 0.3
    lr = 1e-2

    batch_size = 32
    N = 1000

    writer = SummaryWriter()

    num_episodes = 50
    layer_size = 20

    V = nn.Sequential(nn.Linear(3, layer_size), nn.ReLU(), nn.Linear(layer_size, 1))
    optim = torch.optim.Adam(V.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    env = PendulumEnv()
    memory = deque(maxlen=N)
    loss_count = 0

    for ep_num in tqdm(range(num_episodes)):
        state, reward, done, info = env.reset()
        total_episode_reward = reward

        while not done:
            prev_state = state

            # Epsilon-greedy policy during training
            action = (
                np.random.choice(env.action_space)
                if np.random.random() < epsilon
                else act_greedily(V, state, gamma, writer, 400 * ep_num + env.num_steps_taken)
            )

            state, reward, done, info = env.step(action)

            memory.append((prev_state, reward, state, done))

            if batch_size <= len(memory):
                # Randomly sample from memory
                sampled_mem = [
                    memory[idx]
                    for idx in np.random.choice(range(len(memory)), size=batch_size, replace=False)
                ]
                prev_states_batch = torch.stack([idx[0] for idx in sampled_mem])
                reward_batch = torch.unsqueeze(
                    torch.tensor([idx[1] for idx in sampled_mem], dtype=torch.float32), dim=1
                )
                successor_states = torch.stack([idx[2] for idx in sampled_mem])

                # Calculate loss on mini-batch
                prev_state_value = V(prev_states_batch)
                with torch.no_grad():  # This stops us updating
                    td_target = reward_batch + gamma * V(successor_states)

                loss = loss_fn(prev_state_value, td_target)

                # Make the update
                optim.zero_grad()
                loss.backward()
                optim.step()

                # Logging
                writer.add_scalar("Loss", loss, loss_count)
                loss_count += 1

            total_episode_reward += reward

        print("Total episode reward", total_episode_reward)
        writer.add_scalar("Total episode reward", total_episode_reward, ep_num)

        if total_episode_reward > -300:
            break

    # Wrap up tensorboard writer
    writer.flush()
    writer.close()

    save_network(V)


if __name__ == "__main__":
    # train()

    v = load_network()
    validate(lambda x: act_greedily(v, x, gamma))

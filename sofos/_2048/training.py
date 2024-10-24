import math
import os
import pathlib
import random
import re
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sofos._2048 import get_env, get_model
from sofos.device import get_device
from sofos.replay_memory import ReplayMemory, Transition

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means
#           a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.10
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

OFFICIAL_EVALUATIONS_DURATION = 100

# Ensures we can place the matplotlib window where we want
matplotlib.use("Qt5Agg")
# This one is to place the pygame window
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (800, 100)

CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
PATH_TRAINING = CURRENT_PATH / "training_saves"


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_model_version_from_filename(filename: str) -> int:
    match = re.search(r"training_save_v(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        raise RuntimeError(f"Cannot find the version number in {filename=}.")


class Trainer:

    def __init__(
        self,
        version: int,
        display_gym: bool = False,
        save_checkpoints: bool = True,
    ):
        self.version = version
        self.device = get_device()
        self.env = get_env(
            version=version, device=self.device, display_game=display_gym
        )

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        # Get the number of observations
        grid_shape = self.env.observation_space.shape
        n_observations = grid_shape[0] * grid_shape[1]

        ModelClass = get_model(version)
        self.policy_net = ModelClass(n_observations, n_actions).to(self.device)

        self.target_net = ModelClass(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=LR, amsgrad=True
        )
        self.memory = ReplayMemory(10000)

        self.save_checkpoints = save_checkpoints

        self.steps_done = 0
        self.start_epoch = 0

        self.episode_score: list[int] = []

    def select_action(self, state: torch.tensor) -> torch.tensor:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                policy_result = self.policy_net(state)
                largest_column_value = policy_result.max(1)
                max_result = largest_column_value.indices
                action = max_result.view(1, 1)
                return action
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]],
                device=self.device,
                dtype=torch.long,
            )

    def plot_durations(self, show_result=False):
        plt.figure(1)
        scores_t = torch.tensor(self.episode_score, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.plot(scores_t.numpy(), label="Score")
        # Take 100 episode averages and plot them too
        if len(scores_t) >= OFFICIAL_EVALUATIONS_DURATION:
            scores_means = torch.cat(
                (
                    torch.zeros(OFFICIAL_EVALUATIONS_DURATION - 1),
                    (
                        scores_t.unfold(0, OFFICIAL_EVALUATIONS_DURATION, 1)
                        .mean(1)
                        .view(-1)
                    ),
                )
            )
            plt.plot(scores_means.numpy(), label="Mean of scores (last 100)")

        plt.legend(loc="upper left")
        plt.pause(0.1)  # pause a bit so that plots are updated

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043
        # for detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been
        # taken for each batch state according to policy_net
        policy_results = self.policy_net(state_batch)
        state_action_values = policy_results.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with
        # max(1).values This is merged based on the mask, such that we'll
        # have either the expected state value or 0 in case the state was
        # final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA
        ) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(
            PATH_TRAINING / filename,
            map_location=self.device,
            weights_only=False,
        )

        version = extract_model_version_from_filename(filename)
        if version != self.version:
            raise RuntimeError(
                "The version of the file you're trying to load does not match "
                "with the current version configured in the Trainer."
            )

        self.start_epoch = checkpoint["epoch"]
        self.steps_done = checkpoint["steps_done"]
        self.episode_score = checkpoint["episode_score"]
        self.memory = checkpoint["memory"]
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def checkpoint(self, epoch):
        if not self.save_checkpoints:
            return

        average_score = mean(self.episode_score)
        ensure_path(PATH_TRAINING)
        torch.save(
            {
                "epoch": epoch,
                "steps_done": self.steps_done,
                "episode_score": self.episode_score,
                "memory": self.memory,
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            PATH_TRAINING
            / f"training_save_v{self.version}_{epoch}_{average_score}.pt",
        )

        torch.save(
            self.policy_net.state_dict(),
            PATH_TRAINING
            / f"policy_network_v{self.version}_{epoch}_{average_score}.pt",
        )

    def run(self):
        # Enables interactive mode of matplotlib
        plt.ion()
        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(50, 100, 640, 545)

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 1_000_000
        else:
            num_episodes = 50

        i_episode = 0
        try:
            for i_episode in range(self.start_epoch, num_episodes):
                # Initialize the environment and get its state
                state = self.env.reset()
                while True:
                    action = self.select_action(state)
                    next_state, reward, done, info = self.env.step(action)

                    score = info["score"]

                    if done:
                        next_state = None

                    # Store the transition in memory
                    self.memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization
                    # (on the policy network)
                    self.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ)θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[
                            key
                        ] * TAU + target_net_state_dict[key] * (1 - TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                    if done:
                        self.episode_score.append(score)
                        self.plot_durations()
                        break
                if i_episode % 1_000 == 0:
                    self.checkpoint(epoch=i_episode)
        finally:
            self.checkpoint(epoch=i_episode)

        print("Complete")
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    trainer = Trainer(version=2, display_gym=True, save_checkpoints=True)

    # trainer.load_checkpoint("my_file")

    trainer.run()

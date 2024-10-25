import os
import pathlib
from dataclasses import dataclass
from statistics import mean
from typing import Optional

import torch

from sofos.replay_memory import ReplayMemory

CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
PATH_TRAINING = CURRENT_PATH / "training_saves"


@dataclass
class CheckpointData:
    current_epoch: int
    steps_done: int
    episode_durations: list[int]
    memory: ReplayMemory
    model_state_dict: dict
    target_state_dict: dict
    optimizer_state_dict: dict


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint_data(checkpoint_data: CheckpointData):
    average_score = mean(checkpoint_data.episode_durations)
    ensure_path(PATH_TRAINING)

    training_save_filename = (
        f"training_save"
        f"_{checkpoint_data.current_epoch}"
        f"_{average_score}.pt"
    )
    torch.save(
        {
            "epoch": checkpoint_data.current_epoch,
            "steps_done": checkpoint_data.steps_done,
            "episode_durations": checkpoint_data.episode_durations,
            "memory": checkpoint_data.memory,
            "model_state_dict": checkpoint_data.model_state_dict,
            "target_state_dict": checkpoint_data.target_state_dict,
            "optimizer_state_dict": checkpoint_data.optimizer_state_dict,
        },
        PATH_TRAINING / training_save_filename,
    )

    # Also save the policy model separately
    policy_network_filename = (
        f"policy_network"
        f"_{checkpoint_data.current_epoch}"
        f"_{average_score}.pt"
    )
    torch.save(
        checkpoint_data.model_state_dict,
        PATH_TRAINING / policy_network_filename,
    )


def load_checkpoint_data(
    filename: str,
    sub_folder: Optional[str] = None,
    map_location: Optional[str] = None,
) -> CheckpointData:
    path = PATH_TRAINING
    if sub_folder:
        path = path / sub_folder

    checkpoint = torch.load(
        path / filename,
        map_location=map_location,
        weights_only=False,
    )

    return CheckpointData(
        current_epoch=checkpoint["epoch"],
        steps_done=checkpoint["steps_done"],
        episode_durations=checkpoint["episode_durations"],
        memory=checkpoint["memory"],
        model_state_dict=checkpoint["model_state_dict"],
        target_state_dict=checkpoint["target_state_dict"],
        optimizer_state_dict=checkpoint["optimizer_state_dict"],
    )


@dataclass
class PolicyNetworkData:
    model_state_dict: dict


def load_policy_network_data(
    filename: str,
    map_location: str,
    sub_folder: Optional[str] = None,
) -> PolicyNetworkData:
    path = PATH_TRAINING
    if sub_folder:
        path = path / sub_folder

    model_state_dict = torch.load(
        path / filename, map_location=map_location, weights_only=False
    )

    return PolicyNetworkData(model_state_dict=model_state_dict)

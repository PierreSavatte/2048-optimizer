import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Optional

import torch

from sofos._2048 import PATH_TRAINING
from sofos.replay_memory import ReplayMemory


@dataclass
class CheckpointData:
    current_epoch: int
    steps_done: int
    episode_score: list[int]
    memory: ReplayMemory
    model_state_dict: dict
    target_state_dict: dict
    optimizer_state_dict: dict


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_model_version_from_training_save_filename(filename: str) -> int:
    match = re.search(r"training_save_v(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        raise RuntimeError(f"Cannot find the version number in {filename=}.")


def extract_model_version_from_policy_network_filename(filename: str) -> int:
    match = re.search(r"policy_network_v(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        raise RuntimeError(f"Cannot find the version number in {filename=}.")


def save_checkpoint_data(version: int, checkpoint_data: CheckpointData):
    average_score = mean(checkpoint_data.episode_score)
    ensure_path(PATH_TRAINING)

    training_save_filename = (
        f"training_save"
        f"_v{version}"
        f"_{checkpoint_data.current_epoch}"
        f"_{average_score}.pt"
    )
    torch.save(
        {
            "epoch": checkpoint_data.current_epoch,
            "steps_done": checkpoint_data.steps_done,
            "episode_score": checkpoint_data.episode_score,
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
        f"_v{version}"
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
    expected_version: Optional[int] = None,
) -> CheckpointData:
    path = PATH_TRAINING
    if sub_folder:
        path = path / sub_folder

    checkpoint = torch.load(
        path / filename,
        map_location=map_location,
        weights_only=False,
    )

    if expected_version is not None:
        version = extract_model_version_from_training_save_filename(filename)
        if version != expected_version:
            raise RuntimeError(
                "The version of the file you're trying to load does not match "
                "with the current version configured in the Trainer."
            )

    return CheckpointData(
        current_epoch=checkpoint["epoch"],
        steps_done=checkpoint["steps_done"],
        episode_score=checkpoint["episode_score"],
        memory=checkpoint["memory"],
        model_state_dict=checkpoint["model_state_dict"],
        target_state_dict=checkpoint["target_state_dict"],
        optimizer_state_dict=checkpoint["optimizer_state_dict"],
    )


@dataclass
class PolicyNetworkData:
    version: int
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
    version = extract_model_version_from_policy_network_filename(filename)

    return PolicyNetworkData(
        version=version, model_state_dict=model_state_dict
    )

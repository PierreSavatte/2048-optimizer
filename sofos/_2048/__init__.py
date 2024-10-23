from typing import Type

from sofos._2048.environment import (
    Environment,
    ObservableBoard,
    ObservableBoardV1,
    ObservableBoardV2,
)
from sofos._2048.neural_network import DQN, DQNv1, DQNv2

VERSIONS = {
    1: {
        "model": DQNv1,
        "env_board": ObservableBoardV1,
    },
    2: {
        "model": DQNv2,
        "env_board": ObservableBoardV2,
    },
}


def get_env_board(version: int) -> Type[ObservableBoard]:
    return VERSIONS[version]["env_board"]


def get_model(version: int) -> Type[DQN]:
    return VERSIONS[version]["model"]


def get_env(version: int, device: str, display_game: bool = False):
    render_mode = None
    if display_game:
        render_mode = "human"
    return Environment(
        version=version,
        device=device,
        render_mode=render_mode,
    )

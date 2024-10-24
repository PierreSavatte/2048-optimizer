import pathlib
from typing import Type

from sofos._2048.environment import (
    Environment,
    ObservableBoard,
    ObservableBoardV1,
    ObservableBoardV2,
)
from sofos._2048.neural_network import DQN, DQNv1, DQNv2

CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
PATH_TRAINING = CURRENT_PATH / "training_saves"

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
    env_board: Type[ObservableBoard]
    env_board = VERSIONS[version]["env_board"]  # type:ignore
    return env_board


def get_model(version: int) -> Type[DQN]:
    model: Type[DQN] = VERSIONS[version]["model"]  # type:ignore
    return model


def get_env(version: int, device: str, display_game: bool = False):
    render_mode = None
    if display_game:
        render_mode = "human"
    return Environment(
        version=version,
        device=device,
        render_mode=render_mode,
    )

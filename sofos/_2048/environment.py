import math
import random
from abc import ABC, abstractmethod
from typing import Optional

import gymnasium
import numpy as np
import pygame
import torch
from gymnasium import spaces
from torch import Tensor

from sofos._2048.game.classes import Board, Move
from sofos._2048.game.gui import Game

MODEL_MOVE_MAP = {
    0: Move.UP,
    1: Move.DOWN,
    2: Move.LEFT,
    3: Move.RIGHT,
}


def get_move_from_model(value: int) -> Move:
    return MODEL_MOVE_MAP[value]


class ObservableBoard(ABC, Board):

    def __init__(self, *args, device: str, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

    @classmethod
    def get_initial_board(cls, device: str):
        board = cls(device=device)
        board.add_random_tiles(2)
        return board

    @abstractmethod
    def observe(self) -> Tensor: ...


class ObservableBoardV1(ObservableBoard):

    def observe(self) -> Tensor:
        observed_grid = torch.tensor(
            [
                [
                    0 if tile is None else tile.power
                    for line in self.grid
                    for tile in line
                ]
            ],
            device=self.device,
            dtype=torch.float,
        )
        return observed_grid


class ObservableBoardV2(ObservableBoard):

    def observe(self) -> Tensor:
        observed_grid = torch.tensor(
            [
                [0 if tile is None else tile.power for tile in line]
                for line in self.grid
            ],
            device=self.device,
            dtype=torch.float,
        )
        return observed_grid.unsqueeze(0).unsqueeze(0)


class Environment(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        version: int,
        device: str,
        render_mode: Optional[str] = None,
    ):
        from sofos._2048 import get_env_board

        BoardClass = get_env_board(version)

        self.device = device

        # Initialise internal variables
        self.BoardClass = BoardClass
        self.board: ObservableBoard = BoardClass.get_initial_board(
            device=device
        )
        self.previous_merge_count: int = 0
        self.previous_score: int = 0
        self.board_size = self.board.size

        self.render_mode = render_mode
        self.game = None
        if render_mode == "human":
            pygame.init()
            self.game = Game()

        # Initialise gymnasium variables
        self.action_space = spaces.Discrete(len(MODEL_MOVE_MAP))
        self.observation_space = spaces.Box(
            0, 1, (self.board_size, self.board_size), dtype=np.integer
        )
        self.illegal_move_reward = -math.inf
        self.reward_range = (
            self.illegal_move_reward,
            65_536,  # I'd be happy to have a 65_536 tile already!
        )

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    @staticmethod
    def compute_positioning_score(
        tiles_position_map: dict[int, list[tuple[int, int]]]
    ) -> int:
        interesting_keys = sorted(set(tiles_position_map.keys()) - {1, 2})
        if not interesting_keys:
            return 10
        # Only keep the two largest tile sizes
        interesting_keys = interesting_keys[:2]

        score = 0
        for tile_power in interesting_keys:
            for tile_position in tiles_position_map[tile_power]:
                x, y = tile_position
                if (x == 0 or x == 3) or (y == 0 or y == 3):
                    # If the AI placed the largest tiles in the border
                    score += 10
                else:
                    score -= 5

        return score

    @staticmethod
    def compute_duplicated_large_tile_score(
        tiles_position_map: dict[int, list[tuple[int, int]]]
    ):
        interesting_keys = sorted(set(tiles_position_map.keys()) - {1, 2})
        if not interesting_keys:
            return 10
        # Only keep the two largest tile sizes
        interesting_keys = interesting_keys[:2]

        score = 0
        for tile_power in interesting_keys:
            number_of_same_tiles = len(tiles_position_map[tile_power])
            if number_of_same_tiles > 1:
                score -= number_of_same_tiles * 5
            else:
                score += 10

        return score

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        # Extract value from tensor
        action = action.item()

        # Checks that the move is valid
        is_action_valid = self.action_space.contains(action)
        if not is_action_valid:
            raise RuntimeError(f"{action!r} ({type(action)}) invalid")

        move = get_move_from_model(action)
        info = {
            "illegal_move": False,
        }

        move_result = self.board.make_move(move)
        self.render()

        if move_result:
            tiles_position_map = self.board.get_tiles_position_map()
            reward = (
                (self.board.score - self.previous_score)
                + (self.board.merge_count - self.previous_merge_count) * 10
                + (self.board.get_nb_empty_cells()) * 10
                + self.compute_positioning_score(tiles_position_map)
                + self.compute_duplicated_large_tile_score(tiles_position_map)
            )
            self.board.add_random_tiles(1)
            done = self.board.is_game_over()
            if done:
                reward = self.illegal_move_reward
        else:
            reward = self.illegal_move_reward
            info["illegal_move"] = True
            done = False

        self.previous_score = self.board.score
        self.previous_merge_count = self.board.merge_count

        info["score"] = self.board.score

        return (
            self.board.observe(),
            torch.tensor([reward], device=self.device, dtype=torch.float),
            torch.tensor([done], device=self.device, dtype=torch.float),
            info,
        )

    def render(self):
        if self.render_mode is None:
            return
        elif self.render_mode == "human" and self.game is not None:
            self.game.update_tiles(Game.convert_grid(self.board.grid))
            self.game.draw_tiles()
            pygame.display.flip()

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tensor:
        self.seed(seed)
        self.board = self.BoardClass.get_initial_board(device=self.device)
        self.previous_score = 0
        self.previous_merge_count = 0
        return self.board.observe()

    def seed(self, seed: Optional[int] = None):
        random.seed(seed)

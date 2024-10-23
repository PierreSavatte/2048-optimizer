import random
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


class ObservableBoard(Board):

    def __init__(self, *args, device: str, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

    @classmethod
    def get_initial_board(cls, device: str):
        board = cls(device=device)
        board.add_random_tiles(2)
        return board

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


class Environment(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, device: str, render_mode: Optional[str] = None):
        self.device = device

        # Initialise internal variables
        self.board: ObservableBoard = ObservableBoard.get_initial_board(
            device=device
        )
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
        self.illegal_move_reward = -1
        self.reward_range = (
            self.illegal_move_reward,
            65_536,  # I'd be happy to have a 65_536 tile already!
        )

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

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

        if move_result:
            self.board.add_random_tiles(1)

            reward = self.previous_score - self.board.score
            done = self.board.is_game_over()
        else:
            reward = self.illegal_move_reward
            info["illegal_move"] = True
            done = True

        self.render()

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
        self.board = ObservableBoard.get_initial_board(device=self.device)
        self.previous_score = 0
        return self.board.observe()

    def seed(self, seed: Optional[int] = None):
        random.seed(seed)

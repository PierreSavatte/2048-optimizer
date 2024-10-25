from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from tqdm import tqdm

from sofos._2048 import get_env, get_model
from sofos._2048.checkpoint import load_policy_network_data
from sofos.device import get_device

EVALUATION_DURATION = 1_000  # Number of games played


class EndOfTheGameReason(Enum):
    ILLEGAL_MOVE = "ILLEGAL_MOVE"
    BLOCKED = "BLOCKED"


@dataclass
class Evaluation:
    final_grid: list[list[int]]
    final_score: int
    end_of_the_game_reason: EndOfTheGameReason


class BaseEvaluator(ABC):

    def __init__(self, version: int, display_gym: bool = False):
        self.device = get_device()
        self.version = version
        self.env = get_env(
            version=version, device=self.device, display_game=display_gym
        )

    @abstractmethod
    def select_action(self, state: torch.tensor) -> torch.tensor: ...

    @staticmethod
    def get_end_of_the_game(
        done: bool, info: dict
    ) -> Optional[EndOfTheGameReason]:
        if info.get("illegal_move") is True:
            return EndOfTheGameReason.ILLEGAL_MOVE
        if done:
            return EndOfTheGameReason.BLOCKED
        return None

    def run(self) -> list[Evaluation]:
        evaluation_list: list[Evaluation] = []
        print("Starting the evaluation...")
        for _ in tqdm(range(EVALUATION_DURATION)):
            # Initialize the environment and get its state
            state = self.env.reset()
            while True:
                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                end_of_the_game = self.get_end_of_the_game(done, info)

                if end_of_the_game is not None:
                    grid = state.squeeze().tolist()
                    evaluation_list.append(
                        Evaluation(
                            final_grid=grid,
                            final_score=info["score"],
                            end_of_the_game_reason=end_of_the_game,
                        )
                    )
                    break

                state = next_state

        print("Evaluation completed")
        return evaluation_list


class ModelEvaluator(BaseEvaluator):

    def __init__(
        self,
        policy_network_filename: str,
        sub_folder: Optional[str] = None,
        display_gym: bool = False,
    ):
        policy_network_data = load_policy_network_data(
            policy_network_filename,
            sub_folder=sub_folder,
            map_location=get_device(),
        )
        version = policy_network_data.version

        super().__init__(version=version, display_gym=display_gym)

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        # Get the number of observations
        grid_shape = self.env.observation_space.shape
        n_observations = grid_shape[0] * grid_shape[1]

        ModelClass = get_model(version)
        self.policy_net = ModelClass(n_observations, n_actions).to(self.device)

    def select_action(self, state: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_result = self.policy_net(state)
            largest_column_value = policy_result.max(1)
            max_result = largest_column_value.indices
            action = max_result.view(1, 1)
            return action


class RandomStrategyEvaluator(BaseEvaluator):

    def select_action(self, state: torch.tensor) -> torch.tensor:
        return torch.tensor(
            [[self.env.action_space.sample()]],
            device=self.device,
            dtype=torch.long,
        )


def print_model_evaluation(
    policy_network_filename: str,
    display_gym: bool,
    sub_folder: Optional[str] = None,
):
    from sofos._2048.evaluation.metrics import compute_metrics, display_metrics

    model_evaluator = ModelEvaluator(
        policy_network_filename, sub_folder=sub_folder, display_gym=display_gym
    )
    model_evaluation_list = model_evaluator.run()

    random_strategy_evaluator = RandomStrategyEvaluator(
        version=model_evaluator.version, display_gym=display_gym
    )
    random_strategy_evaluation_list = random_strategy_evaluator.run()

    for strategy_name, evaluation_results in [
        (policy_network_filename, model_evaluation_list),
        ("Random Strategy", random_strategy_evaluation_list),
    ]:
        metrics = compute_metrics(evaluation_results)
        display_metrics(metrics, strategy_name=strategy_name)


if __name__ == "__main__":
    print_model_evaluation(
        "policy_network_v2_5000_57.65340795522686.pt",
        sub_folder="with_stop_when_illegal_move",
        display_gym=True,
    )

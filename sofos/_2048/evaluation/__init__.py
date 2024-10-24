from dataclasses import dataclass

import torch
from tqdm import tqdm

from sofos._2048 import get_env, get_model
from sofos._2048.checkpoint import load_policy_network_data
from sofos.device import get_device

EVALUATION_DURATION = 1_000  # Number of games played


@dataclass
class Evaluation:
    final_grid: list[list[int]]
    final_score: int


class Evaluator:

    def __init__(
        self, policy_network_filename: str, display_gym: bool = False
    ):
        self.device = get_device()
        policy_network_data = load_policy_network_data(
            policy_network_filename, map_location=self.device
        )

        self.version = policy_network_data.version
        self.env = get_env(
            version=self.version, device=self.device, display_game=display_gym
        )

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        # Get the number of observations
        grid_shape = self.env.observation_space.shape
        n_observations = grid_shape[0] * grid_shape[1]

        ModelClass = get_model(self.version)
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

    def run(self) -> list[Evaluation]:
        evaluation_list: list[Evaluation] = []
        print("Starting of the evaluation...")
        for _ in tqdm(range(EVALUATION_DURATION)):
            # Initialize the environment and get its state
            state = self.env.reset()
            while True:
                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                if done or info.get("illegal_move"):
                    grid = next_state.tolist()[0][0]
                    evaluation_list.append(
                        Evaluation(final_grid=grid, final_score=info["score"])
                    )
                    break

        print("Evaluation completed")
        return evaluation_list


def print_model_evaluation(policy_network_filename: str, display_gym: bool):
    from sofos._2048.evaluation.metrics import compute_metrics, display_metrics

    evaluator = Evaluator(policy_network_filename, display_gym=display_gym)
    evaluation_list = evaluator.run()

    metrics = compute_metrics(evaluation_list)
    display_metrics(metrics)


if __name__ == "__main__":
    print_model_evaluation(
        "policy_network_v2_5000_1131.3714514194323.pt", display_gym=True
    )

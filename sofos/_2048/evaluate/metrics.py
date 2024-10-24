import statistics
from collections import defaultdict

from sofos._2048.evaluate import Evaluation

Metrics = dict[str, float]

TilesCount = dict[int, int]

TILES_ORDER = [int(2**i) for i in range(1, 200)]


def get_count_of_tiles(grid: list[list[int]]) -> TilesCount:
    tiles_count = defaultdict(int)  # type:ignore
    for row in grid:
        for tile_power in row:
            if tile_power is not None:
                tile_value = int(2**tile_power)
                tiles_count[tile_value] += 1
    return dict(tiles_count)


def get_max_tile_value(grid: list[list[int]]) -> int:
    return max(int(2**tile_power) for row in grid for tile_power in row)


def get_present_tiles(max_tile_value: int) -> list[int]:
    tiles_order_index = TILES_ORDER.index(max_tile_value)
    return TILES_ORDER[: tiles_order_index + 1]


def compute_metrics(evaluations: list[Evaluation]) -> Metrics:
    print("Computing metrics...")
    final_grids = [evaluation.final_grid for evaluation in evaluations]

    present_tile_values = defaultdict(int)  # type:ignore
    number_games_ended = len(evaluations)
    for final_grid in final_grids:
        max_tile_value = get_max_tile_value(final_grid)
        present_tile_values_in_final_grid = get_present_tiles(max_tile_value)
        for present_tile_value in present_tile_values_in_final_grid:
            present_tile_values[present_tile_value] += 1

    metrics = {
        "Number of games ended (stored in memory)": float(number_games_ended)
    }
    for (
        tile_value,
        number_of_games_with_that_tile,
    ) in present_tile_values.items():
        metrics[f"% of games with {tile_value}"] = (
            number_of_games_with_that_tile / number_games_ended
        ) * 100

    metrics["Mean score"] = statistics.mean(
        [evaluation.final_score for evaluation in evaluations]
    )

    return metrics


def display_metrics(metrics: Metrics):
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

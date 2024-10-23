from sofos._2048.environment import Environment


def get_env(device: str, display_game: bool = False):
    render_mode = None
    if display_game:
        render_mode = "human"
    return Environment(device=device, render_mode=render_mode)

import gymnasium as gym


def get_env(display_game: bool = False):
    kwargs = {}
    if display_game:
        kwargs["render_mode"] = "human"
    return gym.make("CartPole-v1", **kwargs)

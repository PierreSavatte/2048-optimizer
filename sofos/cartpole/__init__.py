import gymnasium as gym


def get_env():
    return gym.make("CartPole-v1")

"""Environment wrappers and Gymnasium registrations."""

from envs.crafter_env import CrafterEnv, register_crafter_envs, split_crafter_done

register_crafter_envs()

__all__ = ["CrafterEnv", "register_crafter_envs", "split_crafter_done"]

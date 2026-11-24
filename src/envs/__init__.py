"""Environment wrappers and Gymnasium registrations."""

from envs.crafter_env import CrafterEnv, register_crafter_envs, split_crafter_done
from envs.room_env import RoomEnv
from envs.room_teacher import RoomTeacher

register_crafter_envs()

__all__ = [
    "CrafterEnv",
    "RoomEnv",
    "RoomTeacher",
    "register_crafter_envs",
    "split_crafter_done",
]

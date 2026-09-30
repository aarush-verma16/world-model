import sys
import time

sys.path.insert(0, "src")

from envs.crafter_env import CrafterEnv
from training.crafter_teacher import seed_teacher_episodes
from training.replay_buffer import ReplayBuffer

env = CrafterEnv()
buf = ReplayBuffer(seed=0, max_steps=200_000)
t0 = time.time()
stats = seed_teacher_episodes(env, buf, episodes=6, max_episode_steps=1500, seed=5000, version=2)
dt = time.time() - t0
print("seconds", round(dt, 1), "steps/s", round(stats["steps"] / dt, 1))
print("mean_length", stats["mean_length"])
for k, v in sorted(stats["unlocks"].items()):
    print(f"  {k}: {v}/{stats['episodes']}")
batch = buf.sample(2, 16, teacher_fraction=1.0)
print({k: tuple(v.shape) for k, v in batch.items()})
print("teacher mean", float(batch["teacher"].mean()), "has_local", float(batch["has_local"].mean()))

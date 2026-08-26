"""Plot M4 reward vs lambda-return and copy key stills."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(r"c:\Users\Aarus\Desktop\world-model")
h = json.loads((root / "results/m4_actor_critic/train_metrics.json").read_text(encoding="utf-8"))
steps = [x["step"] for x in h]
rew = [x["reward"] for x in h]
ret = [x["return"] for x in h]
val = [x["value"] for x in h]
ent = [x["entropy"] for x in h]
crit = [x["critic"] for x in h]

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), sharex=True)
axes[0, 0].plot(steps, ret, color="#c45c26", lw=0.9, label="λ-return")
axes[0, 0].plot(steps, val, color="#1f4e79", lw=0.8, alpha=0.85, label="critic value")
axes[0, 0].set_ylabel("imagined")
axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 0].set_title("λ-return tracks the critic, not reward")

axes[0, 1].plot(steps, rew, color="#1b1b1b", lw=0.9)
axes[0, 1].set_ylabel("mean imagined reward / step")
axes[0, 1].set_title("Reward head stays near zero")
axes[0, 1].set_ylim(-0.005, 0.08)

axes[1, 0].plot(steps, ent, color="#1f4e79", lw=0.9)
axes[1, 0].axhline(0.1, color="#c45c26", ls=":", lw=1)
axes[1, 0].set_ylabel("nats")
axes[1, 0].set_xlabel("AC step")
axes[1, 0].set_title("Policy entropy (dotted = 0.1 collapse bar)")

axes[1, 1].plot(steps, crit, color="#1b1b1b", lw=0.9)
axes[1, 1].set_ylabel("two-hot NLL")
axes[1, 1].set_xlabel("AC step")
axes[1, 1].set_title("Critic loss")

for ax in axes.ravel():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 20000)

fig.tight_layout()
out = root / "research paper/figures/m4_return_vs_reward.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print("wrote", out)

fig_dir = root / "research paper/figures"
src = root / "results/m4_actor_critic"
pairs = [
    ("imagine_final.png", "m4_imagine_final.png"),
    ("imagine_step_000200.png", "m4_imagine_step_200.png"),
    ("imagine_step_018000.png", "m4_imagine_step_18000.png"),
    ("imagine_final.gif", "m4_imagine_final.gif"),
]
for a, b in pairs:
    shutil.copy2(src / a, fig_dir / b)
    print("copied", b)

# Experiment Log

Living log of approaches tried, including failures and why they failed.

## Setup (2026-07-31)

- Created conda env `worldmodel` (Python 3.11).
- Confirmed `torch.backends.mps.is_available() == True`.
- Confirmed `CrafterReward-v1` resets/steps with observation shape `(64, 64, 3)`.

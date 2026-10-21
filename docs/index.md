# Fieldcraft

Fieldcraft is a Dreamer agent that crafts inside the tiles it can see. The world model is DreamerV3's RSSM. The actor also reads a 9×7 symbolic map, and is behavior-cloned from a teacher that places the crafting station on those tiles.

- Plan and conventions: `PROJECT_BRIEF.md`
- Experiment log: `experiments.md` (M6–M19 are the Dreamer baseline; M20 onward is Fieldcraft)
- Findings archive for the paper: `research paper/README.md`
- Milestone gates: `milestones.md`

Target runtime is Windows + CUDA (RTX 5080).

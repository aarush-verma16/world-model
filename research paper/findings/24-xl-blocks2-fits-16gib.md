# Finding 24 — Table B.1 CNN `blocks=2` fits 16 GiB at XL seq 32

**Status:** measured on a one-step CUDA smoke, 2026-08-31  
**Kind:** workstation constraint; not a Crafter score and not 14.5  
**Evidence:** `python scripts/count_params.py --smoke --size xl_b2`; `configs/sizes/dreamer_xl_b2.yaml`; finding 04, 18, 20

## Claim

DreamerV3 Table B.1 residual **`blocks=2`** on our XL CNN (channels 96-768, GRU **2560**, seq **32**, batch **16**, `start_mode=all`, bf16) **fits** this 16 GiB card. It is not the VRAM reason to keep `blocks: 1`. The 16 GiB cut in `configs/sizes/dreamer_xl.yaml` was conservative after finding 18’s unused-graph paging, not a measured OOM on the current reinforce graph.

Do not read this as “blocks=2 will print 14.5.” It is the one leftover **model** knob vs Table B.1 that does not require recoding the RSSM (finding 20). M13 (`configs/m13_xl_r512_b2.yaml`) is that experiment at the same `train_ratio` 512 as M12. Do not load M12 weights into it.

## Numbers

| | XL `blocks=1` (`dreamer_xl.yaml`) | XL `blocks=2` (`dreamer_xl_b2.yaml`) |
|---|---|---|
| WM params | **198.0M** | **215.6M** (+17.6M) |
| actor+critic | 16.0M | 16.0M |
| total | 214.1M | 231.7M |
| smoke batch × seq | — | 16 × 32, `start_mode=all` |
| smoke peak VRAM | — | **11.31 / 11.70 GiB** |
| smoke wall | — | 14.14 s for one WM+AC step |

Paper Table B.1 still differs on GRU **4096** (ours 2560 ≈ 200M on this LN-GRU + 2-layer prior; 4096 is ~336M here) and seq **64**. Those stay separate smokes. Jumping blocks + 4096 + seq 64 + ratio 512 in one yaml is the four-cause OOM finding 20 already forbids.

## Failed alternatives

- Rewriting the encoder/RSSM/decoder because M12 last-200 sat at ~1.4 at 100k. The trainer was healthy (`ac_H` ~0.5, `recon_l1` ~0.003). That is the sleep island (findings 21–23), not an untrainable CNN.
- Flipping `blocks: 1` inside `dreamer_xl.yaml` while M12 is still the live run. New size file, new dirs.
- Dropping batch 16→8 because nvidia-smi showed 11 GiB while M12 was also on the card. The isolated smoke peak is 11.3 GiB; compositor already uses ~1.5 GiB; 16 is the default until a real M13 step OOMs.
- Treating `blocks=2` as a new loss or a hunger/combat bonus.

## Paper spin

Compute appendix next to findings 08 / 18: after the unused reinforce graph was removed, Table B.1 `cnn_blocks=2` is cheap on 16 GiB at seq 32 (~18M extra weights, peak ~11 GiB). The remaining XL disagreements are GRU width and sequence length, not “residuals do not fit.”

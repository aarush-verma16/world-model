# The 0.5 env/s was a full card in a long process; lives die in combat

**Claim.** At env 60,928 the live process was at 0.52 env/s. A fresh process on that same checkpoint does 0.87 env/s at 2910 MHz and about 150 W, with 15.8 of 16.3 GiB in use. Half of the actor lives since env 50,000 die before step 200. That is zombie damage, not the hunger clock and not the 10k cap.

## Why this is not the paper

0.87 env/s is two cycles on one checkpoint, not a 20k-step rate. Health assist is the teacher taking the step. Those lives are not the actor's length or the actor's iron pickaxe.

## Evidence

`results/m22_field_craft` log, 2026-10-23: 1.06 env/s at env 54,016, 0.74 through the late 50ks, 0.52–0.53 from 60,160 to 60,928. `vram_alloc` on the log stays 13.79 because it is the peak since process start, not the current reserve.

Fresh process, `ckpt_latest.pt` at env 60,992, batch 8, seq 64, benchmark off, two cycles: 0.82 then 0.87 env/s. World-model step 0.72 s, actor-critic step about 0.44 s. Reserved 14.7 GiB. `nvidia-smi` during the step: 2910 MHz, 141–176 W, 15,869 MiB of 16,303. Teacher v4 `act` is 0.33 ms, so the long assisted lives are not the rate.

`empty_cache` drops reserved to 5.7 GiB and the next cycle is 0.90 env/s. Checkpointing the encoder is 0.80 s/step. Batch 4 is 0.82 s/step. Both are slower than the 0.72 s batch-8 step. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` warns that it is not supported on this Windows build.

Actor lives with env_steps > 50,000: 36 lives, mean length 224, 18 under 200. Wake up 89%, defeat zombie 78%, eat cow 53%. Assisted lives in that stretch include lengths 404, 490, 816. The 816 keeps going after the iron pickaxe because the cap is 10,000 and the actor plays the rest inside an episode tagged teacher.

## What changed

The teacher also takes the step when health is 3 or below. That was turned off after the first 2k steps of the Saturday resume: 11 lives tagged, 0 iron pickaxes. The life ends 32 steps after the iron pickaxe, or after 200 idle teacher steps, instead of running toward 10,000. Half the cloning rows are ordinary teacher-life slices again (eat, sleep, fight). The other half still end on the craft button. Each train cycle returns unused CUDA blocks to the allocator. That does not make a fresh process faster than 0.87.

## Failed alternatives

`expandable_segments` on this box. Encoder checkpointing. Dropping the batch from 8 to 4. Treating the teacher action as the cost.

## Paper spin

On a 16 GiB card the XL batch-8 seq-64 step is memory-bandwidth bound near 150 W, and a collect length near 200 is the combat hit, not a truncated episode.

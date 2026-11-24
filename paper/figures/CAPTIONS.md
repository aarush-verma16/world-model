# Figure captions

Files in this directory are the only images the draft includes. Captions below are the claims those images can carry.

## `room-crop-facing-station.png`

The 4×4 crop from the cell south of the station, facing it. Each tile is 16×16 pixels in the 64×64 frame the encoder sees; the file is that frame scaled with nearest-neighbor. The station is the orange tile in the nearest row. This is the environment, not a reconstruction.

## `room-crop-facing-away.png`

The same cell, facing away. The station is one cell behind the agent and is absent from the image. `use` still scores from this pose when the run requires it, and it does not score from this pose when the run requires the station to be in the crop.

## `field-open-loop-station-inside.png`

Open-loop strip from the 100k checkpoint of the run with the station inside the crop and the loss window anchored. The top row is context. The field-head numbers in `evidence.md` are the result. The imagined frames are dark and low-contrast; that is not evidence the posterior decoder is empty.

## `field-open-loop-window-off.png`

The same strip for the matched run with the loss window off.

## `field-open-loop-station-outside.png`

The same strip for the matched run whose teacher walks the station outside the crop.

The three strips agree with each other. Do not rank the runs by how gray the imagined row looks.

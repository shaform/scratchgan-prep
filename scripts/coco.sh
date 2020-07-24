#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="train" \
  --dataset=coco-raw \
  --data_dir=data/coco-raw \
  --checkpoint_dir=data/coco-raw/checkpoints/

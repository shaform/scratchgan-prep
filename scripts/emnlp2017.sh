#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="train" \
  --dataset=emnlp2017-raw \
  --data_dir=data/emnlp2017-raw \
  --checkpoint_dir=data/emnlp2017-raw/checkpoints/

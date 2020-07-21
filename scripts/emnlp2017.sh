#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="train" \
  --dataset=emnlp2017-prep \
  --data_dir=data/emnlp2017-prep \
  --checkpoint_dir=data/emnlp2017-prep/checkpoints/

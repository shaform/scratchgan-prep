#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="evaluate_pair" \
  --dataset=emnlp2017-raw \
  --data_dir=data/emnlp2017-raw \
  --checkpoint_dir=data/emnlp2017-raw/checkpoints/ \
  --num_examples_for_eval=10000

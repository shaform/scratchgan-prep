#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="evaluate_pair" \
  --dataset=emnlp2017-prep \
  --data_dir=data/emnlp2017-prep \
  --checkpoint_dir=data/emnlp2017-prep/checkpoints-cfg/ \
  --trainable_embedding_size=512 \
  --num_examples_for_eval=10000

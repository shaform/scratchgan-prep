#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="evaluate_ppl" \
  --dataset=emnlp2017-prep \
  --data_dir=data/emnlp2017-prep \
  --checkpoint_dir=data/emnlp2017-prep/checkpoints-cfg/ \
  --trainable_embedding_size=512

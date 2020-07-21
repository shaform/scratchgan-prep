#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="train" \
  --dataset=coco-prep \
  --data_dir=data/coco-prep \
  --checkpoint_dir=data/coco-prep/checkpoints-cfg/ \
  --trainable_embedding_size=512

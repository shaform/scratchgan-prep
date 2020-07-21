#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="evaluate_pair" \
  --dataset=coco-prep \
  --data_dir=data/coco-prep \
  --checkpoint_dir=data/coco-prep/checkpoints-cfg/ \
  --batch_size=500 \
  --num_examples_for_eval=500 \
  --trainable_embedding_size=512

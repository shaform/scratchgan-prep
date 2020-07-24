#!/bin/bash
python3 -m scratchgan.experiment \
  --mode="evaluate_ppl" \
  --dataset=coco-raw \
  --data_dir=data/coco-raw \
  --checkpoint_dir=data/coco-raw/checkpoints/ \
  --batch_size=500 \
  --num_examples_for_eval=500

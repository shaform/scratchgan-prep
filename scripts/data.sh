#!/bin/bash

EMNLP=data/emnlp2017-raw
COCO=data/coco-raw


mkdir -p $EMNLP $COCO

for split in test; do
  if [ ! -f $COCO/$split.txt ]; then
    curl https://raw.githubusercontent.com/pclucas14/GansFallingShort/bb0fe64604a6c49d220a2bf232cc2c69f6846ec8/real_data_experiments/data/coco/$split.txt --output $COCO/$split.txt
  fi
done
if [ ! -f $COCO/train-raw.txt ]; then
    curl https://raw.githubusercontent.com/pclucas14/GansFallingShort/bb0fe64604a6c49d220a2bf232cc2c69f6846ec8/real_data_experiments/data/coco/train.txt --output $COCO/train-raw.txt
fi
if [ ! -f $COCO/valid.txt ]; then
  python scripts/split_valid.py $COCO/train-raw.txt $COCO/train.txt $COCO/valid.txt
fi
for split in train valid test; do
  if [ ! -f $EMNLP/$split.txt ]; then
    curl https://raw.githubusercontent.com/pclucas14/GansFallingShort/bb0fe64604a6c49d220a2bf232cc2c69f6846ec8/real_data_experiments/data/news/$split.txt --output $EMNLP/$split.txt
  fi
done

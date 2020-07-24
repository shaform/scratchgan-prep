#!/bin/bash

EMNLP=data/emnlp2017-raw
COCO=data/coco-raw


mkdir -p $EMNLP $COCO

for split in train valid test; do
  if [ ! -f $COCO/$split.txt ]; then
    curl https://raw.githubusercontent.com/pclucas14/GansFallingShort/bb0fe64604a6c49d220a2bf232cc2c69f6846ec8/real_data_experiments/data/coco/$split.txt --output $COCO/$split.txt
  fi
done
for split in train valid test; do
  if [ ! -f $EMNLP/$split.txt ]; then
    curl https://raw.githubusercontent.com/pclucas14/GansFallingShort/bb0fe64604a6c49d220a2bf232cc2c69f6846ec8/real_data_experiments/data/news/$split.txt --output $EMNLP/$split.txt
  fi
done

#!/bin/bash
# python train_class_split.py --fold 0 --gpu 0

python inference.py --image_folder "$1" --scene_ids "$2" --output "$3"

# python train_class_split.py --fold 8 --gpu 3
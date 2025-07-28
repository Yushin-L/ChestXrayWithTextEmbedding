#!/bin/bash

# Loop through seed values from 1 to 5
for N in {1..5}
do
    # echo "Running accelerate launch with seed=$N"
    # CUDA_VISIBLE_DEVICES=2 python /home/imyousin12/vit/lih_vit_embed_valid.py --seed $N
    sleep 5
    python /home/ubuntu/mimic_cxr/cls_valid.py --seed $N 
    sleep 5

done
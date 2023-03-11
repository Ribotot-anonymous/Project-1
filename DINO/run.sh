#!/bin/bash

python main_train.py \
--save_path exp/ECAPATDNN512_exp1 \
--batch_size 112 \
--n_mels 80 \
--nOut 256 \
--test_interval 1 \
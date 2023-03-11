#!/bin/bash

python main_train_new_.py \
--save_path exp/ECAPATDNN512_exp \
--batch_size 96 \
--n_mels 80 \
--nOut 256 \
--label_ratio 7 \
--test_interval 1 \
--freeze_last_layer 1 \

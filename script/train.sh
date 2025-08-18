#!/bin/bash
# Can be train integratively here, or can be trained command by command separately.
# Please make sure all the preproces are done before training. 

## Parameter ####
vq_idx=1
cuda=0
exp_id=1
#################

python3 train_vqvae.py --vq_idx $vq_idx --data_type target --cuda $cuda
# python3 train_vqva.py --vq_idx $vq_idx --data_type others --cuda $cuda
python3 token_extract.py --cuda $cuda --vq_idx $vq_idx --data_type target --ckpt_dir ckpt/ --mel_dir data/mel --output_dir data/token
python3 token_extract.py --cuda $cuda --vq_idx $vq_idx --data_type others --ckpt_dir ckpt/ --mel_dir data/mel --output_dir data/token
python3 train_lm.py --cuda $cuda --exp_idx $exp_id

###############
# python3 train_vqvae.py --vq_idx 1 --data_type target --cuda 0
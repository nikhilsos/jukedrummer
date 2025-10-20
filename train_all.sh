!/bin/bash
# Can be train integratively here, or can be trained command by command separately.
# Please make sure all the preproces are done before training. 
export PYTHONPATH="/home/nikhil/jukedrummer/:$PYTHONPATH" 
## Parameter ####
vq_idx=1
cuda=0
exp_id=1
#################

python3 train_vqvae.py --vq_idx $vq_idx --data_type target --cuda $cuda
python3 train_vqvae.py --vq_idx $vq_idx --data_type others --cuda $cuda
python3 token_extract.py --cuda $cuda --vq_idx $vq_idx --data_type target --ckpt_dir ckpt/ --mel_dir data/mel --output_dir data/token
python3 token_extract.py --cuda $cuda --vq_idx $vq_idx --data_type others --ckpt_dir ckpt/ --mel_dir data/mel --output_dir data/token
python3 train_lm.py --cuda $cuda --exp_idx $exp_id

# # clear older fine tuning data
# rm -r /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/wavs/* 

# python extract_vqvae_recon_mel.py

# # copy data for new fine tuning


# fine tune on the data reconstructed by vqvae
# python3 hifi_gan/train.py \
#     --config /home/nikhil/jukedrummer/hifi_gan/config_v1.json \
#     --input_wavs_dir /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/wavs\
#     --input_training_file /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/training.txt \
#     --input_validation_file /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/validation.txt \
#     --checkpoint_path /home/nikhil/jukedrummer/hifi_gan/cp_hifigan \
#     --cuda 0 \
#     --input_mels_dir /home/nikhil/jukedrummer/hifi_gan/ft_dataset/target \
#     --fine_tuning True

###############

# copy the generator to jukedrummer checkpoint directory

# python3 train.py \
#     --config config_v1.json \
#     --input_wavs_dir /home/lego/NAS189/home/codify/data/drums/hop_audio_24s/wavs \
#     --input_training_file /home/lego/NAS189/home/codify/data/drums/feature/finetune_mel/melgan/train_files.txt \
#     --input_validation_file /home/lego/NAS189/home/codify/data/drums/feature/finetune_mel/melgan/test_files.txt \
#     --checkpoint_path /home/lego/NAS189/home/codify/ckpt/hifigan/ \
#     --cuda 0 \
#     --input_mels_dir /home/lego/NAS189/home/codify/data/drums/feature/finetune_mel/melgan \
#     --fine_tuning true
export PYTHONPATH="/home/nikhil/jukedrummer/:$PYTHONPATH" 

python3 train.py \
    --config /home/nikhil/jukedrummer/hifi_gan/config_v1.json \
    --input_wavs_dir /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/wavs \
    --input_training_file /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/training.txt \
    --input_validation_file /home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/validation.txt \
    --checkpoint_path /home/nikhil/jukedrummer/hifi_gan/cp_hifigan/ \
    --cuda 0 \
    --input_mels_dir /home/nikhil/jukedrummer/data/audio/reconstructed_mel_recon_vq1\
    --fine_tuning True
    

    # --input_mels_dir /home/nikhil/jukedrummer/hifi_gan/ft_dataset/target \

    # --input_mels_dir /home/nikhil/jukedrummer/hifi_gan/ft_dataset \
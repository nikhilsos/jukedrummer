
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
    --input_wavs_dir /home/nikhil/jukedrummer/data/segment_audio/target/ \
    --input_training_file /home/nikhil/jukedrummer/data/segment_audio/training.txt \
    --input_validation_file /home/nikhil/jukedrummer/data/segment_audio/validation.txt \
    --checkpoint_path /home/nikhil/jukedrummer/hifi_gan/cp_hifigan_pansori/ \
    --cuda 1 \
    --input_mels_dir /home/nikhil/jukedrummer/data/mel_from_tokens_vq1/ \
    --fine_tuning False
    # --input_mels_dir /home/nikhil/jukedrummer/hifi_gan/ft_dataset/target \
    # --input_mels_dir /home/nikhil/jukedrummer/hifi_gan/ft_dataset \
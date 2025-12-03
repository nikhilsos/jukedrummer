#!/bin/bash

python3 utils/preprocess.py --audio_dir data/audio/ --segment_audio_dir data/segment_audio --mel_dir data/mel/ --beat_dir data/token --cuda 0 --segment_by_downbeats False


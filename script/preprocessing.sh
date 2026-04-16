#!/bin/bash
python3 utils/preprocess.py --audio_dir data/audio/ --segment_audio_dir data/segment_audio --mel_dir data/mel/ --cuda 0 #--segment_by_downbeats
python3 extract_dphase_tokens.py


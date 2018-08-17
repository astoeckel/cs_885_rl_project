#!/bin/sh

./gym_speech_resynthesis/gym_speech_resynthesis/envs/control_network.py \
	--training-data-dir ./data/control_sequences_small/ \
	--n-epochs 4096 --n-dims "$1" \
	--trace ./data/control_seq_n_dim_"$1"_trace.npy \
	--tar ./data/control_seq_n_dim_"$1"_weights.h5

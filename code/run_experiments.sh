#!/usr/bin/bash

PIDS=

kill_children() {
    echo "Killing child processes..."
    trap - SIGINT SIGTERM
    kill $PIDS
}

trap kill_children SIGINT SIGTERM

run_single() {
	if [ $2 = "1" ]; then
		DATASET="data/training_data_control_seq"
	elif [ $2 = "2" ]; then
		DATASET="data/training_data"
	else
		die "Invalid second argument to run_single!"
	fi
	python3 \
		-m "gym_speech_resynthesis.run_$1" \
                --mfcc-window 0 \
		--target-dir "data/experiments/out_dataset_$2_$1_trial_$3" \
		--training-data-dir $DATASET &
	PIDS="$PIDS $!"
}

TRIAL=2

module load mpi

run_single acktr 1 $TRIAL
run_single ddpg 1 $TRIAL
run_single ppo2 1 $TRIAL
run_single trpo 1 $TRIAL

run_single acktr 2 $TRIAL
run_single ddpg 2 $TRIAL
run_single ppo2 2 $TRIAL
run_single trpo 2 $TRIAL

wait


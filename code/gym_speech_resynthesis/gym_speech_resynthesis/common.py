#!/usr/bin/env python3

#   Copyright (C) 2018  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

def parse_args(cback=None):
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--seed',
        type=int,
        default=0)
    parser.add_argument(
        '--num-timesteps',
        type=int,
        default=10000000)
    parser.add_argument(
        '--training-data-dir',
        type=str,
        default='data/training_data_control_seq/')
    parser.add_argument(
        '--target-dir',
        type=str,
        default='data/synth_out/')
    parser.add_argument(
        '--control-network-weight-file',
        type=str,
        default='data/control_net/control_seq_n_dim_1_weights_014.h5')
    parser.add_argument(
        '--mfcc-window',
        type=int,
        default=2)
    if not cback is None:
        cback(parser)
    args = parser.parse_args()
    return args

def make_env(args):
    import numpy as np
    from gym_speech_resynthesis.envs import SpeechResynthesisEnvMFCC

    return SpeechResynthesisEnvMFCC(
        control_network_weight_file = args.control_network_weight_file,
        training_data_dir = args.training_data_dir,
        target_dir = args.target_dir,
        mfcc_window = args.mfcc_window
    )

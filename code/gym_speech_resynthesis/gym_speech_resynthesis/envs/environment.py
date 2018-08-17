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

from copy import copy
import h5py
import numpy as np
import scipy.io.wavfile
import os

import gnuspeech_trm

from . import mfcc, audio, control_network

FS = 16000  # Sample rate


def discretise_len(x):
    """
    Used to discretise a training sample lenght in seconds into the
    corresponding bin index.
    """
    return int(x * 10)



class TrainingChunk:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_training_chunks(training_data_dir):
    min_avail_episode_len, max_avail_episode_len = np.inf, -np.inf
    max_n_chunks = 0
    chunks = {}
    for root, _, files in os.walk(training_data_dir):
        for file in files:
            if not file.lower().endswith('.h5'):
                continue
            path = os.path.join(root, file)
            with h5py.File(path, 'r') as f:
                chunk_len = discretise_len(f['len'][0])
                max_avail_episode_len = max(f['len'][0], max_avail_episode_len)
                min_avail_episode_len = min(f['len'][0], min_avail_episode_len)
                max_n_chunks = max(max_n_chunks, f['mfcc'].shape[0])
                if not chunk_len in chunks:
                    chunks[chunk_len] = []
                chunks[chunk_len].append(path)
    return chunks, min_avail_episode_len, max_avail_episode_len, max_n_chunks

def read_traning_chunk(fn):
    """
    Reads the training data chunk with the given filename and returns it.
    """
    with h5py.File(fn, 'r') as f:
        return TrainingChunk(
            in_file=str(f['in_file'][0]),
            out_file=str(f['out_file'][0]),
            fs=int(f['fs'][0]),
            len=f['len'][0],
            n=f['mfcc'].shape[0],
            mfcc=np.array(f['mfcc']),
            spectrum=np.array(f['spectrum']),
            pcm=np.array(f['pcm']),
            ts=np.array(f['ts']))

class TrainingRecord:
    def __init__(self):
        self.mfcc_synth = []
        self.spectrum_synth = []
        self.mfcc_orig = []
        self.spectrum_orig = []
        self.samples = []
        self.action = []
        self.reward = []

    def dump(self, f):
        with h5py.File(f, 'w') as f:

            def S(name, data):
                f.create_dataset(name, data=data, compression="gzip")

            S('mfcc_synth', np.array(self.mfcc_synth, dtype=np.float32))
            S('spectrum_synth', np.array(
                self.spectrum_synth, dtype=np.float32))
            S('mfcc_orig', np.array(self.mfcc_orig, dtype=np.float32))
            S('spectrum_orig', np.array(self.spectrum_orig, dtype=np.float32))
            S('samples',
              np.array(
                  np.clip(self.samples, -1, 1) * ((1 << 15) - 1),
                  dtype=np.float32))
            S('action', np.array(self.action, dtype=np.float32))
            S('reward', np.array(self.reward, dtype=np.float32))


class Environment:
    """
    The Environment class implements the overall environment the RL agent
    operates in. It provides the observations, receives the agent's action and
    computes the reward.
    """

    def _pick_training_chunk(self, max_episode_len):
        # Limit the max_episode_len to the maximum episode length available
        max_episode_len = max(self.min_avail_episode_len,
                              min(self.max_avail_episode_len, max_episode_len))

        # Convert the length to the discretised version
        ilen = discretise_len(max_episode_len)

        # Randomly select a bin
        bins = list(filter(lambda i: i <= ilen, self.chunks.keys()))
        bin_ = self.chunks[bins[self.random.randint(len(bins))]]

        # Randomly select a training sample from the bin
        return bin_[self.random.randint(len(bin_))]

    def __init__(self,
                 training_data_dir,
                 control_network_weight_file,
                 mfcc_window=2,
                 random_state=None,
                 target_dir=None,
                 record_interval=20):
        """
        Initializes the environment. Indexes the training data stored in
        "training_data_dir".

        training_data_dir: directory where the training data chunks generated by
                           generate_training_data.py are stored. Recursively
                           scans this directory.
        control_network_weight_file:
                           file containing the neural network definition of the
                           control network.
        mfcc_window:       number of future and past MFCC vectors that are
                           presented to the agent.
        random_state:      numpy random state instance that should be used.
        target_dir:        directory in which the output files are stored.
        record_interval:   controls how often a complete training record is
                           stored.
        """

        print("Creating environment instance...")

        # Generate the random_state instance if we have none
        if random_state is None:
            random_state = np.random
        self.random = random_state

        # Load the control network
        self.control_network = control_network.ControlNetwork(
            control_network_weight_file)

        # Copy some flags
        self.target_dir = target_dir
        self.record_interval = max(1, int(record_interval))

        # Open the traget trace file
        self.trace_file = None
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            self.trace_file = open(os.path.join(target_dir, 'trace.csv'), 'w')

        # Scan the training data directory and index the training data by len
        self.chunks, self.min_avail_episode_len, self.max_avail_episode_len, \
            self.max_n_chunks = read_training_chunks(training_data_dir)

        # Create the TRM model
        self.trm = gnuspeech_trm.TRM()

        # Create the MFCC analysis instance used to analyse the output of the
        # TRM
        self.mfcc = mfcc.MFCCFeatureAnalysis(sample_rate_in=FS, sample_rate=FS)

        # Copy the window size, compute the observation size
        assert (mfcc_window >= 0)
        self.action_size = self.control_network.n_dims
        self.mfcc_window = int(mfcc_window)

        M, W, S = self.mfcc.mfcc_size, self.mfcc_window, self.trm.total_params
        self.observation_size = M * (2 * W + 1)

        # Ensure that we start with a fresh episode
        self.done = True
        self.training_chunk = None
        self.training_record = None
        self.t = self.mfcc_window
        self.episode_index = 0
        self.episode_sample = 0
        self.total_sample = 0

    def reset(self, max_episode_len=np.inf):
        """
        Ends the current episode, resets the TRM and the MFCC calculation.
        """

        # Dump the synthesized audio to disk
        if self.training_chunk and self.target_dir and self.training_record and \
                len(self.training_record.samples) > 0:
            # Make sure the target directory exists
            TC = self.training_chunk
            tar_dir = os.path.join(self.target_dir,
                                   '{:03d}'.format(discretise_len(TC.len)))
            os.makedirs(tar_dir, exist_ok=True)

            # Write the data to disk
            idx = self.episode_index
            tar_fn = os.path.join(tar_dir, '{}_{:04d}.h5'.format(
                os.path.splitext(os.path.basename(TC.out_file))[0], idx))
            self.training_record.dump(tar_fn)

            # Write the WAV files to disk
            tar_wav = os.path.join(tar_dir, '{}_{:04d}.wav'.format(
                os.path.splitext(os.path.basename(TC.out_file))[0], idx))
            tar_wav_orig = os.path.join(tar_dir, '{}_orig.wav'.format(
                os.path.splitext(os.path.basename(TC.out_file))[0], idx))
            scipy.io.wavfile.write(
                tar_wav, FS,
                np.array(
                    np.clip(
                        np.concatenate(self.training_record.samples), -1, 1) *
                    ((1 << 15) - 1),
                    dtype=np.int16))
            if not os.path.exists(tar_wav_orig):
                scipy.io.wavfile.write(tar_wav_orig, FS, TC.pcm)

        # Flush the trace file if one is open
        if self.trace_file:
            self.trace_file.flush()

        # Reset the recorder
        if self.episode_index % self.record_interval == 0:
            self.training_record = TrainingRecord()
        else:
            self.training_record = None
        self.samples = None
        self.episode_sample = 0

        # Reset the TRM and the MFCC analysis
        self.trm.reset()
        self.trm.configure(gnuspeech_trm.TRM.voice_male)
        self.trm.reference_glottal_pitch = 0.0
        self.trm.filter_period = 20e-3
        self.trm.output_rate = FS
        self.mfcc.reset()

        # Pick a new training sample
        self.done = False
        chunk_fn = self._pick_training_chunk(max_episode_len)
        self.training_chunk = read_traning_chunk(chunk_fn)
        self.t = self.mfcc_window  # Start at time t=W to be able to look back

        return self.training_chunk

    def observe(self):
        """
        Returns the current observation. If the environment is not in an episode
        at the moment, starts a new episode with the given length.

        Returns two values: a flag indicating whether the agent is currently at
        the end of an episode or not, and the current observation. The
        observation vector has lenght `observation_size`.

        max_avail_episode_len: maximum length of a new episode in seconds. The
        environment will pick an episode length between the minimum and maximum
        length with uniform probability.
        target_logger: Python logger instance to log to. Using own logger
        instance for OpenAI Gym Environment
        """
        if self.done:
            return True, np.zeros(self.observation_size)

        # Some handy aliases
        t, W = self.t, self.mfcc_window

        # Fetch the observation vector, scale observation to [-1, 1]
        TC = self.training_chunk
        observation = np.concatenate(TC.mfcc[(t - W):(t + W + 1)])
        observation = np.clip(observation / 10, -1, 1)

        # Go to the next index in time
        self.t += 1

        # Update the "done" flag, return the observation
        done = self.t >= self.training_chunk.n - W
        if done != self.done:
            self.episode_index += 1
            self.done = done
        return self.done, observation

    def act(self, action):
        """
        Receives the action from the agent and returns the reward. The action is
        a vector of length `action_size`.

        action: continuous vector containing the action of the agent.
        regularisation: factor by which the magnitude of the action vector
        counts as negative reward.
        """
        if self.done or np.any(np.isnan(action)):
            return 0.0  # We're done, do nothing

        actionIn = action
        action, _ = self.control_network.eval(actionIn)
        self.trm.set_parameters(action)

        # Render the next few samples
        sample_count = self.mfcc.fft_size // self.mfcc.oversample
        samples = self.trm.synthesize(sample_count)
        self.samples = samples

        # Feed the samples into the MFCC analysis
        mfccs, spectrum, ts = self.mfcc(samples)

        # Ignore wnd MFCCs by shifting the resulting timestamps
        ts += self.mfcc_window * self.mfcc.smpl_wnd / self.mfcc.sample_rate_in

        # Compute the reward "earned" by matching the synthesized speech to the
        # reference signal
        reward_total = 0
        TC = self.training_chunk
        for i, t in enumerate(ts):
            j = np.argmin(np.abs(t - self.training_chunk.ts))

            m_orig = TC.mfcc[j]
            m_synth = mfccs[i]
            r = -np.sqrt(np.mean((m_orig - m_synth)**2))

            reward_total += r

            # Record data to the training record, if we currently have one
            if not self.training_record is None:
                self.training_record.mfcc_synth.append(mfccs[i])
                self.training_record.mfcc_orig.append(TC.mfcc[j])
                self.training_record.spectrum_synth.append(spectrum[i])
                self.training_record.spectrum_orig.append(TC.spectrum[j])
                self.training_record.reward.append(r)

        # Record more data to the training record, if we currently have one
        if not self.training_record is None:
            self.training_record.action.append(actionIn)
            self.training_record.samples.append(
                np.pad(samples, (0, sample_count - len(samples)), 'constant'))

        # Record the training trace
        if not self.trace_file is None:
            self.trace_file.write(
                '{},{},{},{}\n'.format(self.total_sample, self.episode_sample,
                                       self.episode_index, reward_total))
        self.total_sample += 1
        self.episode_sample += 1

        return reward_total


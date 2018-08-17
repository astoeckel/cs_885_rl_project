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
"""
This program reads an audio file containing pure speech and extracts chunks of a
few seconds.
"""

import numpy as np
import subprocess


def from_file(filename, channels=1, sample_rate=16000):
    """
    Uses FFMpeg to load audio and convert it to RAW data at the given channel
    count and rate. Returns a numpy array containing the data with samples
    in rows and individual channel data in columns.
    """
    proc = subprocess.Popen([
       'ffmpeg',                 # Start ffmpeg
       '-i',   filename,         # Pass the function argument as first parameter
       '-vn',                    # Discard video streams
       '-c:a', 'pcm_f32le',      # Use 32-bit float as a codec (little endian)
       '-f',   'f32le',          # Use 32-bit float in raw mode as format
       '-ac',  str(channels),    # Output the given number of channels...
       '-ar',  str(sample_rate), # ...at the given rate
       '-'],                     # Output to stdout
       stdout=subprocess.PIPE,
       stderr=subprocess.DEVNULL)
    data = proc.communicate()[0]
    retval = proc.wait()
    if retval != 0:
        raise Exception(
            "Error while reading audio file, make sure you have a recent " +
            "version of FFMpeg installed!")
    return np.reshape(np.frombuffer(data, dtype='<f4'), (-1, channels))


def extract_chunks(data,
                   sample_rate=16000,
                   threshold=0.05,
                   flt_len=0.5,
                   min_gap_len=0.1,
                   min_chunk_len=0.5,
                   max_chunk_len=10.0,
                   subsample=10):
    """
    Extracts chunks in the input audio.
    """

    # Threshold the input data, subsample
    data_th = np.mean(np.atleast_2d(data), axis=1) > threshold
    data_th = data_th[::subsample]
    N = data.size

    # Filter the tresholded data and threshold again
    flt_w = int(flt_len * sample_rate / subsample)
    flt = np.exp(-0.1 * np.linspace(-1, 1, flt_w)**2)
    flt = flt / np.sum(flt)
    data_th = np.convolve(data_th * 1.0, flt, 'same') > (0.5 * threshold)

    i_min_gap_len = int(min_gap_len * sample_rate / subsample)
    i_min_chunk_len = int(min_chunk_len * sample_rate / subsample)
    i_max_chunk_len = int(max_chunk_len * sample_rate / subsample)
    v_last, res = False, []
    o = {
        'i_on': 0,
        'i_off': 0
    }

    def handle_transition(i, v, is_final=False):
        if v:
            # If this was a long gap, add the last chunk to the result list,
            # otherwise ignore it
            i_gap_len = (i - o['i_off'])
            i_chunk_len = (o['i_off'] - o['i_on'])
            if i_gap_len >= i_min_gap_len or is_final:
                if i_chunk_len >= i_min_chunk_len and i_chunk_len <= i_max_chunk_len:
                    res.append((
                        min(N, o['i_on'] * subsample),
                        min(N,  o['i_off'] * subsample)))
                o['i_on'] = i
        else:
            o['i_off'] = i

    for i, v in enumerate(data_th):
        # Only handle edges in the signal
        if v == v_last:
            continue
        v_last = v
        handle_transition(i, v)

    # Make sure final chunks get reported out
    handle_transition(len(data_th) + 1, False, is_final=True)
    handle_transition(len(data_th) + 2, True, is_final=True)

    return res


class Player:
    """
    Plays back the given RAW audio matrix. Samples must be organised in rows,
    channel information is organised in columns. Uses the "aplay" program which
    ships with ALSA.
    """

    def __init__(self, channels, sample_rate):
        """
        Creates a new Player instance. Use in conjunction with Python's "with
        resources" feature.

        channels: number of channels in the audio data.
        rate: samples per seconds.
        """
        self.channels = channels
        self.rate = sample_rate
        self.proc = subprocess.Popen(
            ["aplay", "-r",
             str(sample_rate), "-c",
             str(channels), "-f", "FLOAT_LE"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.PIPE)

    def write(self, data):
        """
        Sends the given data to the audio player. Blocks until the data has been
        read by the child process (has been played back).

        data: numpy array containing the audio data. Samples must be organised
        in rows, channel information is organised in columns.
        """
        assert (data.shape[1] == self.channels)
        self.proc.stdin.write(memoryview(np.ascontiguousarray(data, "<f4")))
        self.proc.stdin.flush()

    def close(self):
        """
        Waits until all remaining audio data has been played. No additional data
        can be played back.
        """
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


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

import scipy.signal
import scipy.interpolate
import numpy as np

# Note: This code is in part based on the following blog post:
# http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html


class MFCCFeatureAnalysis:
    @staticmethod
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    @staticmethod
    def mel_to_hz(m):
        return 700.0 * ((10.0**(m / 2595.0)) - 1.0)

    @staticmethod
    def build_mel_filterbank(fft_size, mel_bands, fs):
        # Calculate the start/end points of the triangle filters in Hz
        f_max = int(np.floor(fs / 2.0))
        m_max = MFCCFeatureAnalysis.hz_to_mel(f_max)
        m_pnts = np.linspace(0, m_max, mel_bands + 2)
        f_pnts = MFCCFeatureAnalysis.mel_to_hz(m_pnts)

        # Compute the corresponding FFT bins
        bin_max = int(np.floor(fft_size / 2 + 1))
        bins = np.floor(2 * bin_max * f_pnts / fs)

        # Create a triangle window around each FFT bin
        res = np.zeros((mel_bands, bin_max))
        for i in range(1, mel_bands + 1):
            fml, fmc, fmr = int(bins[i - 1]), int(bins[i]), int(bins[i + 1])
            for k in range(fml, fmc):
                res[i - 1, k] = (k - fml) / (fmc - fml)
            for k in range(fmc, fmr):
                res[i - 1, k] = (fmr - k) / (fmr - fmc)
        return res

    def __init__(self,
                 fft_size=512,
                 mfcc_size=12,
                 oversample=4,
                 mel_bands=40,
                 sample_rate_in=44100,
                 sample_rate=16000,
                 bp_min=0.1,
                 bp_max=0.8,
                 bp_taps=41,
                 lift_spectrum=True,
                 lift_mfccs=True):
        # Copy the input parameters
        self.fft_size = fft_size
        self.mfcc_size = mfcc_size
        self.mel_bands = mel_bands
        self.sample_rate_in = sample_rate_in
        self.sample_rate = sample_rate
        self.oversample = oversample
        self.smpl_wnd = fft_size / oversample
        self.lift_spectrum = lift_spectrum
        self.lift_mfccs = lift_mfccs

        # Setup the band-pass filter for downsampling
        self.bp = scipy.signal.firwin(
            bp_taps,
            [bp_min * (sample_rate * 0.5), bp_max * (sample_rate * 0.5)],
            window='blackmanharris',
            fs=sample_rate_in,
            pass_zero=False)

        # Window function
        self.wnd = scipy.signal.windows.blackmanharris(fft_size)

        # Setup the MEL triangle filters
        self.mel_bank = MFCCFeatureAnalysis.build_mel_filterbank(
            fft_size, mel_bands, sample_rate)

        # Setup the ring buffer used for the interpolation
        if sample_rate != sample_rate_in:
            buf_size = int(fft_size * sample_rate_in / sample_rate)
            t_max = buf_size / sample_rate_in
            self.buf_in_ts = np.linspace(0, t_max, buf_size + 1)[:-1]
            self.buf_out_ts = np.linspace(0, t_max, fft_size + 1)[:-1]
            self.buf = np.zeros(buf_size + bp_taps - 1)
        else:
            self.buf = np.zeros(fft_size)
        self.buf_ptr = 0
        self.smpl_counter = 0

        # Compute the lifter
        ms, ns = np.arange(self.mel_bands), np.arange(self.mfcc_size)
        cep_lifter = (self.mfcc_size - 1) * 2
        spe_lifter = (self.mel_bands - 1) * 2
        self.lift_m = 1 + (cep_lifter / 2) * np.sin(np.pi * ns / cep_lifter)
        self.lift_s = 1 + (spe_lifter / 2) * np.sin(np.pi * ms / spe_lifter) * 0.01

    def reset(self):
        """
        Resets the internal buffer pointers to ensure independence of the data
        passed to __call__.
        """
        self.buf_ptr = 0
        self.smpl_counter = 0

    def __call__(self, smpls):
        # Append the given samples to the internal buffer
        res_mfcc, res_spectrum, res_t = [], [], []
        read_ptr, n_in, n_buf = 0, len(smpls), len(self.buf)
        while read_ptr < n_in:
            # Copy samples to the ring buffer
            n_read = min(n_in - read_ptr, n_buf - self.buf_ptr)
            w0, w1 = self.buf_ptr, self.buf_ptr + n_read
            r0, r1 = read_ptr, read_ptr + n_read
            self.buf[w0:w1] = smpls[r0:r1]
            self.buf_ptr, read_ptr = w1, r1
            self.smpl_counter += n_read

            # If the ring-buffer is full, downsample, otherwise abort
            if self.buf_ptr < n_buf:
                break

            # Apply the band-pass and interpolate between the samples
            if self.sample_rate != self.sample_rate_in:
                buf_flt = np.convolve(self.buf, self.bp, 'valid')
                buf_ip = scipy.interpolate.interp1d(
                    self.buf_in_ts, buf_flt, 'quadratic', assume_sorted=True)
                buf = buf_ip(self.buf_out_ts)
            else:
                buf = np.copy(self.buf)

            # Apply the window function to the buffer
            buf *= self.wnd

            # Compute the power spectrum
            fft = np.fft.fft(buf)[:(self.fft_size // 2) + 1]
            spectrum = np.real(np.abs(fft))

            # Apply the MEL filters to the spectrum
            spectrum_mel = self.mel_bank @ spectrum

            # Make sure the zero-coefficients are non-zero, then
            # compute the spectrum in dB and rescale
            spectrum_mel = np.where(spectrum_mel == 0.0,
                                    np.finfo(float).eps, spectrum_mel)
            if self.lift_spectrum:
                spectrum_mel *= self.lift_s
            spectrum_mel_db = 20.0 * np.log10(spectrum_mel)
            spectrum_mel_db_n = np.clip((spectrum_mel_db - 15.0) / 27.0, -1, 1)

            # Compute the Cepstral Coefficients
            mfcc = scipy.fftpack.dct(
                spectrum_mel_db_n, type=2, norm='ortho')[1:(self.mfcc_size + 1)]
            if self.lift_mfccs:
                mfcc *= self.lift_m

            # Append to the result
            res_mfcc.append(mfcc)
            res_spectrum.append(spectrum_mel_db_n)
            res_t.append((self.smpl_counter - 0.5 * self.fft_size) / self.sample_rate)

            # Shift content in the ring buffer to the left
            n_shift = int((self.sample_rate_in / self.sample_rate) *
                          (self.fft_size / self.oversample))
            self.buf_ptr = n_buf - n_shift
            self.buf[:self.buf_ptr] = self.buf[n_buf - self.buf_ptr:]

        return np.array(res_mfcc), np.array(res_spectrum), np.array(res_t)


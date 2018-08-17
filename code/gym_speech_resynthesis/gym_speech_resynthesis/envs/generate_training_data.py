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

FS = 16000  # Sample rate


def process_single_file(in_file, out_dir):
    """
    Reads the given input file, normalises it and splits it into chunks of at
    most ten seconds length. Computes spectrogram and MFCCs. Packs the file into
    a uniquely named HDF5 file in the specified output directory.
    """
    import hashlib
    import numpy as np
    import os
    import h5py

    try:
        import audio
        import mfcc
    except:
        from . import audio
        from . import mfcc

    # Compute the canonical path and the SHA256 hash of the file
    hashobj = hashlib.sha224()
    with open(in_file, 'rb') as f:
        hashobj.update(f.read())

    # Read and normalize the data
    print('{}: Decoding file'.format(in_file))
    data = audio.from_file(in_file, channels=1, sample_rate=FS)
    data = 0.99 * data / np.max(np.abs(data))

    # Compute the chunks
    print('{}: Extracting chunks'.format(in_file))
    chunks = audio.extract_chunks(data, sample_rate=FS)

    # Compute the MFCCs and spectrogram for each chunk
    mfcc_analysis = mfcc.MFCCFeatureAnalysis(sample_rate_in=FS, sample_rate=FS)
    out_files = []
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(chunks)):
        i_start, i_end = chunks[i]
        chunk_len = int((i_end - i_start) / FS * 10)

        # Compute the filename of the chunk
        hashobj2 = hashobj.copy()
        hashobj2.update(i_start.to_bytes(4, byteorder='little'))
        hashobj2.update(i_end.to_bytes(4, byteorder='little'))
        out_file = os.path.join(out_dir, '{:03d}'.format(chunk_len),
                                'chunk_{:03d}_{}.h5'.format(
                                    chunk_len, hashobj2.hexdigest()[:16]))
        out_files.append(out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Process a chunk
        print('{}: Processing chunk {} ({}/{})'.format(in_file, out_file, i +
                                                       1, len(chunks)))

        # Normalize the individual chunk
        chunk = data[i_start:i_end, 0]
        chunk = 0.99 * chunk / np.max(np.abs(chunk))

        # Zero out silence to save space
        flt = np.exp(-10.0 * np.linspace(-1.0, 1.0, int(0.05 * FS))**2)
        flt = flt / np.sum(flt)
        chunk_flt = np.convolve(np.abs(chunk), flt, 'same')
        mask = np.convolve(1.0 * (chunk_flt > 0.02), flt, 'same')
        chunk = chunk * mask
        chunk = 0.99 * chunk / np.max(np.abs(chunk))

        # Compute MFCCs and Spectrum
        mfcc_analysis.reset()
        mfccs, spectrum, ts = mfcc_analysis(chunk)

        # Compute the 16-bit signed chunk data
        chunk16 = (chunk * ((1 << 15) - 16)).astype(np.int16)

        # Store everything in an HDF5 file
        with h5py.File(out_file, "w") as f:
            strtype = h5py.special_dtype(vlen=str)
            f.create_dataset('in_file', (1, ), data=in_file, dtype=strtype)
            f.create_dataset('out_file', (1, ), data=out_file, dtype=strtype)
            f.create_dataset('fs', (1, ), data=np.int32(FS))
            f.create_dataset(
                'len', (1, ), data=np.float32((i_end - i_start) / FS))
            f.create_dataset(
                'mfcc', data=mfccs.astype(np.float32), compression=9)
            f.create_dataset(
                'spectrum', data=spectrum.astype(np.float32), compression=9)
            f.create_dataset('pcm', data=chunk16, compression=9)
            f.create_dataset('ts', data=ts.astype(np.float32), compression=9)

    manifest_file = os.path.join(out_dir,
                                 os.path.basename(in_file) + '.manifest')
    with open(manifest_file, 'w') as f:
        for out_file in out_files:
            f.write(out_file + '\n')

    print('{}: Done'.format(in_file))


def process_dir(in_dir, out_dir):
    import os
    import sys
    import subprocess
    import multiprocessing

    makefile = ''
    all_targets = []
    for root, subdirs, files in os.walk(in_dir):
        for file in sorted(files):
            if file.lower().endswith('.mp3') or file.lower().endswith('.opus'):
                in_file = os.path.join(root, file)
                manifest_file = os.path.join(out_dir, file + '.manifest')
                makefile += manifest_file + ': ' + in_file + '\n'
                makefile += '\t' + sys.argv[0] \
                                 + ' --in-file "' + in_file + '"' \
                                 + ' --out-dir "' + out_dir + '"\n\n'
                all_targets.append(manifest_file)
    makefile = '%PHONY: all\n\nall: ' + ' '.join(
        all_targets) + '\n\n' + makefile

    # Execute the makefile
    j = multiprocessing.cpu_count()
    res = subprocess.run(
        ["make", "-j" + str(j), "-f", "-"],
        input=bytes(makefile, "utf-8"),
        stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise Exception(str(res.stderr, "utf-8"))


# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(
    description=
    'Reads audio files and chunks them into training data for resynthesis')
parser.add_argument(
    '--in-file',
    type=str,
    default='',
    help='Single input file, must not be specified at the same time as ' +
    '--in-dir')
parser.add_argument(
    '--in-dir',
    type=str,
    default='',
    help='Input directory; recursively searches for audio files in this ' +
    'directory. Must not be specified at the same time as --in-file')
parser.add_argument(
    '--out-dir',
    type=str,
    required=True,
    help='Output directory; writes the resulting files to this directory')

args = parser.parse_args()

# Either process a single file or a directory
if bool(args.in_file) == bool(args.in_dir):
    import sys
    print('Must specify exactly one of --in-file or --in-dir!')
    sys.exit(1)
if bool(args.in_file):
    process_single_file(args.in_file, args.out_dir)
else:
    process_dir(args.in_dir, args.out_dir)


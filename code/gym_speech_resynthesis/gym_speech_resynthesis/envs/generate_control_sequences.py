#!/usr/bin/env python3

import argparse
import sys
import os
import multiprocessing
import subprocess
import re
import hashlib
import shlex

parser = argparse.ArgumentParser(
    description='Generates reference control trajectories from GNU Speech SA')
parser.add_argument(
    '--txt',
    type=str,
    required=True,
    help='Textfile containing the sentences that should be generated')
parser.add_argument(
    '--out-dir', type=str, required=True, help='Target directory')
parser.add_argument(
    '--gnuspeech',
    required=True,
    type=str,
    help=
    'Command line used to call gnuspeech; this script will append the -p, -o, and text parameters'
)

args = parser.parse_args()


def count_upper(s):
    return sum(map(lambda c: 1 if c == c.upper() else 0, s))


# Assemble a makefile compiling the input text to motor control sequences, text
# snippets and the corresponding audio
makefile = ''
all_targets = []
with open(args.txt, 'r') as f:
    # Extract sentences from the input
    txt = f.read()

    # Filter ALL CAPS lines (headers in Project Gutenberg books)
    lines = filter(lambda s: (count_upper(s) / (1 + len(s))) < 0.75,
                   txt.split('\n'))

    # Perform some substitutions
    txt = re.sub('\\s+', ' ', ' '.join(lines))
    txt = re.sub('[“”]', '"', txt)

    # Iterrate over sentences
    sentences = map(lambda x: x.strip(), re.split("[.:]", txt))
    sentences = map(lambda x: x + '.', filter(lambda x: len(x) >= 4, sentences))

    # Filter words/sentences crashing gnuspeech
    sentences = filter(lambda x: not 'sqush' in x, sentences)
    sentences = filter(lambda x: len(x) < 800, sentences)

    sentences = list(sentences)
    for i, sentence in enumerate(sentences):

        # Compute the target filenames for the sentence
        hash_ = hashlib.sha224(sentence.encode('utf-8')).hexdigest()[:8]
        tar_dir = os.path.join(args.out_dir, '{:03d}'.format(i // 100))
        tar_wav = os.path.join(tar_dir, '{:04d}_{}.wav'.format(i, hash_))
        tar_txt = os.path.join(tar_dir, '{:04d}_{}.txt'.format(i, hash_))
        tar_opus = os.path.join(tar_dir, '{:04d}_{}.opus'.format(i, hash_))
        tar_params = os.path.join(tar_dir, '{:04d}_{}.prm'.format(i, hash_))
        tar_params_gz = os.path.join(tar_dir, '{:04d}_{}.prm.gz'.format(
            i, hash_))

        # Generate the makefile entry
        makefile += tar_params_gz + ': ' + shlex.quote(args.txt) + '\n'
        makefile += '\t' + '@echo Processing {}/{} \n'.format(
            i + 1, len(sentences))
        makefile += '\t' + '@mkdir -p ' + shlex.quote(tar_dir) + '\n'
        makefile += '\t' + '@' + args.gnuspeech \
                         + ' -p ' + shlex.quote(tar_params) \
                         + ' -o ' + shlex.quote(tar_wav) \
                         + ' ' + shlex.quote(sentence) + ' > /dev/null \n'
        makefile += '\t' + '@echo ' + shlex.quote(sentence) \
                         + ' > ' + shlex.quote(tar_txt) + '\n'
        makefile += '\t' + '@opusenc --bitrate 48' \
                         + ' ' + shlex.quote(tar_wav) \
                         + ' ' + shlex.quote(tar_opus) + '\n'
        makefile += '\t' + '@gzip' \
                         + ' ' + shlex.quote(tar_params) + '\n'
        makefile += '\t' + '@rm ' + shlex.quote(tar_wav) + '\n'
        all_targets.append(tar_params_gz)

makefile = '%PHONY: all\n\nall: ' + ' '.join(all_targets) + '\n\n' + makefile

# Execute the makefile
j = multiprocessing.cpu_count()
res = subprocess.run(
    ["make", "-j" + str(j), "-f", "-"],
    input=bytes(makefile, "utf-8"),
    stderr=subprocess.PIPE)
if res.returncode != 0:
    raise Exception(str(res.stderr, "utf-8"))


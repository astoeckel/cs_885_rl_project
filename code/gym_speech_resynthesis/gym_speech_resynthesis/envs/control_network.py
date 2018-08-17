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

import numpy as np
import h5py


class ControlNetworkAutoencoder:

    learning_rate = 0.1e-2

    n_in = 15
    n_h1 = 32
    n_h2 = 16
    n_h4 = 16
    n_h5 = 32

    def __init__(self, n_dims=4):
        import tensorflow as tf

        # Copy the dimensionality of the auto-encoder bottleneck
        self.n_h3 = n_dims

        # Graph inputs
        self.X = tf.placeholder("float", [None, self.n_in])
        self.Y = tf.placeholder("float", [None, self.n_in])

        # Layer weights and biases
        self.wX1 = tf.Variable(tf.random_normal([self.n_in, self.n_h1]))
        self.w12 = tf.Variable(tf.random_normal([self.n_h1, self.n_h2]))
        self.w23 = tf.Variable(tf.random_normal([self.n_h2, self.n_h3]))
        self.w34 = tf.Variable(tf.random_normal([self.n_h3, self.n_h4]))
        self.w45 = tf.Variable(tf.random_normal([self.n_h4, self.n_h5]))
        self.w5Y = tf.Variable(tf.random_normal([self.n_h5, self.n_in]))
        self.b1 = tf.Variable(tf.random_normal([self.n_h1]))
        self.b2 = tf.Variable(tf.random_normal([self.n_h2]))
        self.b3 = tf.Variable(tf.random_normal([self.n_h3]))
        self.b4 = tf.Variable(tf.random_normal([self.n_h4]))
        self.b5 = tf.Variable(tf.random_normal([self.n_h5]))
        self.bY = tf.Variable(tf.random_normal([self.n_in]))


        # Create the computational graph
        self.n1 = tf.nn.tanh(tf.add(tf.matmul(self.X, self.wX1), self.b1))
        self.n2 = tf.nn.tanh(tf.add(tf.matmul(self.n1, self.w12), self.b2))
        self.n3 = tf.nn.tanh(tf.add(tf.matmul(self.n2, self.w23), self.b3))

        # Add noise to the autoencoding layer to simulate discretisation
        self.n4 = tf.nn.tanh(tf.add(tf.matmul(self.n3, self.w34), self.b4))
        self.n5 = tf.nn.tanh(tf.add(tf.matmul(self.n4, self.w45), self.b5))
        self.Ypred = tf.add(tf.matmul(self.n5, self.w5Y), self.bY)

        # Define the loss function and the optimizer
        self.loss_op = tf.reduce_mean(
            tf.losses.mean_squared_error(
                labels=self.Y, predictions=self.Ypred))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.opt.minimize(self.loss_op)
        self.init = tf.global_variables_initializer()

    def train(self, Xs, n_epochs=10, n_batches=128, batch_size=512, seed=None):
        import tensorflow as tf
        import sys

        np.random.seed(seed)
        trace = np.empty((2, n_epochs))
        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(n_epochs):
                avg_cost = 0
                for i in range(n_batches):
                    batch_X = Xs[np.random.randint(0, Xs.shape[0], batch_size)]
                    _, c = sess.run(
                        [self.train_op, self.loss_op],
                        feed_dict={self.X: batch_X,
                                   self.Y: batch_X})
                    avg_cost += c / n_batches
                trace[0, epoch] = (epoch + 1) * n_batches
                trace[1, epoch] = avg_cost
                sys.stderr.write(
                    "\r{:6.2f}% done; n_dims={}, epoch={:4},. cost={:12.9f}".format(
                        100 * (epoch + 1) / n_epochs, self.n_h3, epoch, avg_cost))
            sys.stderr.write("\nDone training.\n")
            return trace, {
                "n_dims": self.n_h3,
                "wX1": self.wX1.eval(),
                "w12": self.w12.eval(),
                "w23": self.w23.eval(),
                "w34": self.w34.eval(),
                "w45": self.w45.eval(),
                "w5Y": self.w5Y.eval(),
                "b1": self.b1.eval(),
                "b2": self.b2.eval(),
                "b3": self.b3.eval(),
                "b4": self.b4.eval(),
                "b5": self.b5.eval(),
                "bY": self.bY.eval()
            }


class ControlNetwork:

    def __init__(self, weight_file, autoencoder=False):
        """
        Reads the control network parameters from the given HDF5 file.
        """
        self.autoencoder = autoencoder
        self.pitch_min, self.pitch_max = -30.0, 10.0
        self.pitch_offs = -0.5 * (self.pitch_max + self.pitch_min)
        self.pitch_scale = 2.0 / (self.pitch_max - self.pitch_min)
        with h5py.File(weight_file, 'r') as f:
            for key in [
                    'Xoffs', 'Xscale', 'Xmin', 'Xmax', 'n_dims',
                    'wX1', 'w12', 'w23', 'w34', 'w45', 'w5Y',
                    'b1', 'b2', 'b3', 'b4', 'b5', 'bY'
            ]:
                self.__dict__[key] = np.array(f[key])
            self.n_dims = np.array(f['n_dims']) + 1 # Pitch is a seperate channel

    def eval(self, x):
        """
        Evaluates the control network; returns the TRM parameters.

        x: Input vector; length must be equal to self.n_dims
        """
        if self.autoencoder:
            a1 = np.tanh((x[1:] + self.Xoffs) * self.Xscale @ self.wX1 + self.b1)
            a2 = np.tanh(a1 @ self.w12 + self.b2)
            a3 = np.tanh(a2 @ self.w23 + self.b3)
            Yi = np.zeros(self.n_dims)
            Yi[0] = (x[0] + self.pitch_offs) * self.pitch_scale
            Yi[1:] = a3
        else:
            Yi = x
        a4 = np.tanh(Yi[1:] @ self.w34 + self.b4)
        a5 = np.tanh(a4 @ self.w45 + self.b5)
        Y = np.zeros(16)
        Y[1:] = a5 @ self.w5Y + self.bY
        Y[1:] = np.clip(Y[1:] / self.Xscale - self.Xoffs, self.Xmin, self.Xmax)

        # Insert the pitch signal
        Y[0] = np.clip(Yi[0], -1, 1) / self.pitch_scale - self.pitch_offs

        # Suppress small fluctuations near zero on the volume channels
        Y[1:4] = np.clip((Y[1:4] - 0.1) * (60./59.9), 0, 60)

        return Y, Yi

def read_control_sequence(file):
    import gzip

    with gzip.open(file, 'rt') as f:
        txt = f.read()
    lines = txt.split('\n')[25:-1]  # Skip header
    arr = [list(map(float, line.split(' '))) for line in lines]
    return np.array(arr)

def read_control_sequences(path):
    """
    Loads the control trajectories from the files produced by the
    "generate_control_sequences" script.
    """
    import os
    import sys
    trajectories = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            if not file.lower().endswith('.prm.gz'):
                continue
            sys.stderr.write('\rLoading {}...'.format(file))
            trajectories.append(read_control_sequence(os.path.join(root, file)))
    sys.stderr.write("\rDone...                        \n")
    return trajectories

if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description='Trains the control network')
    parser.add_argument(
        '--training-data-dir',
        type=str,
        required=True,
        help='Control trajectories output directory')
    parser.add_argument(
        '--n-dims', type=int, required=True, help='Autoencoder depth')
    parser.add_argument(
        '--seed',
        type=int,
        default=14891,
        help='Random seed for sample selection and perturbation noise')
    parser.add_argument(
        '--n-epochs', type=int, default=1024, help='Number of epochs (128 batches per epoch)')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument(
        '--repeat', type=int, default=16, help='Number of repetitions')
    parser.add_argument(
        '--trace',
        type=str,
        required=True,
        help='Target file documenting the learning process')
    parser.add_argument(
        '--tar',
        type=str,
        required=True,
        help='Target file containing the learned weights')

    args = parser.parse_args()
    trajectories = read_control_sequences(args.training_data_dir)
    print('Loaded {} trajectories with {} samples'.format(
        len(trajectories), sum(map(len, trajectories))))

    X = np.concatenate(trajectories)[:, 1:] # Ignore the pitch component
    Xmin, Xmax = np.min(X, axis=0), np.max(X, axis=0)
    Xoffs = -0.5 * (Xmin + Xmax)
    Xscale = 2.0 / np.maximum(.1, (Xmax - Xmin))
    Xnorm = (X + Xoffs[None, :]) * Xscale[None, :]
    net = ControlNetworkAutoencoder(args.n_dims)
    traces = np.empty((args.repeat + 1, args.n_epochs))
    for i in range(args.repeat):
        trace, weights = net.train(
            Xnorm, n_epochs=args.n_epochs, batch_size=args.batch_size, seed=args.seed+i)
        if i == 0:
            traces[0] = trace[0]
        traces[i + 1] = trace[1]

        # Store the weights and normalisation factors as h5 file
        tar_file_parts = os.path.splitext(args.tar)
        tar_file = '{}_{:03d}{}'.format(tar_file_parts[0], i, tar_file_parts[1])
        with h5py.File(tar_file, 'w') as f:
            f.create_dataset('Xmin', data=Xmin)
            f.create_dataset('Xmax', data=Xmax)
            f.create_dataset('Xoffs', data=Xoffs)
            f.create_dataset('Xscale', data=Xscale)
            for key, entry in weights.items():
                f.create_dataset(key, data=entry)

    # Store the trace as numpy file
    np.save(args.trace, traces, allow_pickle=False)


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

import gym
import gym.spaces as spaces
import gym.logger as logger
import gym.utils.seeding

import numpy as np

from . import audio
from . import environment


class Spec:
    def __init__(self, id_, timestep_limit):
        self.id = id_
        self.timestep_limit = timestep_limit


class SpeechResynthesisEnv(gym.Env):
    """
    Wraps the speech resynthesis environment defined in environment.py in an
    OpenAI-compatible Gym Environment.
    """

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self,
                 id_,
                 control_network_weight_file,
                 training_data_dir,
                 target_dir,
                 mfcc_window=2):
        # Create the actual environment instance
        self.env = environment.Environment(
            control_network_weight_file=control_network_weight_file,
            training_data_dir=training_data_dir,
            target_dir=target_dir,
            mfcc_window=mfcc_window)

        # Maximum path length
        self.spec = Spec(id_, self.env.max_n_chunks + 1)

        # Setup the vectorial action and observation space
        M, N = self.env.observation_size, self.env.action_size
        self.observation_space = spaces.Box(-1, 1, (M, ), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (N, ), dtype=np.float32)
        self.reward_scale = 1.0

        # Audio player
        self.player = None

    def seed(self, seed=None):
        # Set the environment random seed
        self.env.random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Perform the action and gather reward
        reward = self.env.act(action) * self.reward_scale

        # Observe the environment
        done, obs = self.env.observe()

        return obs, reward, done, {}

    def reset(self):
        # Reset the environment, fetch a new traning chunk
        chunk = self.env.reset(max_episode_len=10.0)
        if self.player:
            self.player.write((0.9 * chunk.pcm / (1 << 15)).reshape(-1, 1))

        # Log the name and length of the training chunk
        logger.info('Training on chunk {}, len {:4.2f}'.format(
            chunk.out_file, chunk.len))

        # Fetch and return the first observation
        _, obs = self.env.observe()
        return obs

    def render(self, mode='human'):
        """
        Renders the current synthesiser output to the audio hardware.
        """
        if not mode == 'human':
            return  # Abort if the wrong render mode is given

        # Create the player for audio rendering
        if self.player is None:
            try:
                self.player = audio.Player(
                    channels=1, sample_rate=environment.FS)
            except:
                self.player = None

        # Render the current sample buffer to the player
        if (not self.player is None) and (not self.env.samples is None):
            self.player.write(self.env.samples.reshape(-1, 1))
            self.env.samples = None  # Prevent samples from being written twice

    def close(self):
        """
        Closes the renderer.
        """
        if not self.player is None:
            self.player.close()
        self.player = None


class SpeechResynthesisEnvMFCC(SpeechResynthesisEnv):
    def __init__(self, *args, **kwargs):
        super().__init__("speech-resynthesis-mfcc-v0", *args, **kwargs)


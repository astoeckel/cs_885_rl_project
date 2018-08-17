#!/usr/bin/env python3

from . import common
from baselines import logger


def train(args, num_timesteps, seed):
    import tensorflow as tf

    from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
    from baselines.acktr.acktr_cont import learn
    from baselines.acktr.policies import GaussianMlpPolicy
    from baselines.acktr.value_functions import NeuralNetValueFunction

    env = common.make_env(args)
    env.reward_scale = 0.01

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

    env.close()


if __name__ == '__main__':
    args = common.parse_args()
    logger.configure()
    train(args, num_timesteps=args.num_timesteps, seed=args.seed)


#!/usr/bin/env python3

from . import common
from baselines import bench, logger

def train(args):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import tensorflow as tf

    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    tf.Session(config=config).__enter__()

    def make_env():
        env = common.make_env(args)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(args.seed)
    policy = MlpPolicy
    model = ppo2.learn(
        policy=policy,
        env=env,
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=args.num_timesteps)


if __name__ == '__main__':
    args = common.parse_args()
    logger.configure()
    train(args)


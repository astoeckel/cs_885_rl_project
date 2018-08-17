#!/usr/bin/env python3

from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from mpi4py import MPI

from . import common

def run(args, seed, noise_type, layer_norm, evaluation, **kwargs):
    import time
    import os
    import baselines.ddpg.training as training
    from baselines.ddpg.models import Actor, Critic
    from baselines.ddpg.memory import Memory
    from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

    import tensorflow as tf

    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = common.make_env(args)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = common.make_env(args)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    def add_args(parser):
        boolean_flag(parser, 'render-eval', default=False)
        boolean_flag(parser, 'layer-norm', default=True)
        boolean_flag(parser, 'normalize-returns', default=False)
        boolean_flag(parser, 'normalize-observations', default=True)
        parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
        parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
        parser.add_argument('--actor-lr', type=float, default=1e-4)
        parser.add_argument('--critic-lr', type=float, default=1e-3)
        boolean_flag(parser, 'popart', default=False)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--reward-scale', type=float, default=1.)
        parser.add_argument('--clip-norm', type=float, default=None)
        parser.add_argument('--nb-epoch-cycles', type=int, default=20)
        parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
        parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
        parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
        parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
        boolean_flag(parser, 'evaluation', default=False)

    # Parse arguments, adjust the number of epochs according to the given number
    # of timesteps
    args = common.parse_args(add_args)
    setattr(args, 'nb_epochs', args.num_timesteps // (args.nb_rollout_steps * args.nb_epoch_cycles))
    print('nb_epochs={}'.format(args.nb_epochs))
    setattr(args, 'render', False)
    return args

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Setup logger on first MPI node
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()

    # Run actual script.
    from copy import copy
    args_dict = copy(vars(args))
    del args_dict['num_timesteps']
    del args_dict['training_data_dir']
    del args_dict['target_dir']
    del args_dict['control_network_weight_file']
    del args_dict['mfcc_window']
    run(args, **args_dict)


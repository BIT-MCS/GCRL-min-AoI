import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from method.explorer import Explorer
from policies.policy_factory import policy_factory
from envs.model.agent import Agent


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_random_seeds(args.randomseed)
    args.output_dir=args.model_dir

    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.config is not None:
        config_file = args.config
    else:
        config_file = os.path.join(args.model_dir, 'config.py')  # TODO：注意这里需要改
    model_weights = os.path.join(args.model_dir, 'best_val.pth')
    logging.info('Loaded RL weights with best VAL')

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure environment
    env = gym.make('CrowdSim-v0')
    agent = Agent()
    human_df = env.human_df

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.set_device(device)
    policy.configure(policy_config, human_df)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    policy.set_phase(args.phase)
    policy.set_env(env)
    agent.set_policy(policy)
    agent.print_info()
    env.set_agent(agent)

    explorer = Explorer(env, agent, device, None, gamma=0.9)

    for i in range(10):

        explorer.run_k_episodes(k=1, phase=args.phase, args=args, plot_index=i+1)
        logging.info(f'Testing #{i} finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-m', '--model_dir', type=str, default="logs/debug")
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=0)

    parser.add_argument('--vis_html', default=False, action='store_true')
    parser.add_argument('--plot_loop', default=False, action='store_true')

    # parser.add_argument('-d', '--planning_depth', type=int, default=None)
    # parser.add_argument('-w', '--planning_width', type=int, default=None)
    # parser.add_argument('--sparse_search', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)

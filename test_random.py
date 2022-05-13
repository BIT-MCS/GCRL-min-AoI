import os
import gym
import argparse
import shutil
import importlib
import logging
import torch
import sys

from method.explorer import Explorer
from policies.policy_factory import policy_factory
from envs.model.agent import Agent


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    return None


def main(args):
    set_random_seeds(args.random_seed)
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    # 仅仅知道模块名字和路径的情况下import模块
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)  # 通过传入模块的spec返回新的被导入的模块对象
    spec.loader.exec_module(config)

    # configure logging
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)  # 输出日志信息到磁盘文件
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('Current config content is :{}'.format(config))
    device = torch.device("cpu")

    # configure environment
    env = gym.make('CrowdSim-v0')
    human_df = env.human_df

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    policy.set_device(device)
    policy.configure(policy_config,human_df)



    agent = Agent()
    agent.set_policy(policy)
    env.set_agent(agent)
    explorer = Explorer(env, agent, device,gamma=0.9)

    # random policy
    explorer.run_k_episodes(k=1, phase='test', args=args,plot_index=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/infocom_benchmark/random.py')
    parser.add_argument('--output_dir', type=str, default='logs/debug')
    parser.add_argument('--overwrite', default=False, action='store_true')

    parser.add_argument('--debug', default=False, action='store_true')  # 开启debug模式
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--vis_html', default=False, action='store_true')
    parser.add_argument('--plot_loop', default=False, action='store_true')
    parser.add_argument('--moving_line', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)

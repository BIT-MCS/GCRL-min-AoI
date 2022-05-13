import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy

from torch.utils.tensorboard import SummaryWriter
from envs.model.agent import Agent
from method.trainer import MPRLTrainer
from method.memory import ReplayMemory
from method.explorer import Explorer
from policies.policy_factory import policy_factory


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)  # !!!
    return None


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_random_seeds(args.randomseed)

    # configure paths
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
                exit(0)
    if make_new_dir:
        base_config = os.path.join(os.path.join(os.path.split(args.config)[0], os.pardir), 'config.py')
        os.makedirs(args.output_dir)
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))
        shutil.copy(base_config, os.path.join(args.output_dir, 'base_config.py'))

    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

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
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    if torch.cuda.is_available() and args.gpu:
        logging.info('Using gpu: %s' % args.gpu_id)
    else:
        logging.info('Using device: cpu')

    writer = SummaryWriter(log_dir=args.output_dir)

    # configure environment
    env = gym.make('CrowdSim-v0')
    agent = Agent()
    human_df = env.human_df

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()  # model_predictive_rl
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy.set_device(device)
    policy.configure(policy_config, human_df)

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    num_batches = train_config.train.num_batches
    num_episodes = train_config.train.num_episodes
    sample_episodes = train_config.train.sample_episodes
    warmup_episodes = train_config.train.warmup_episodes
    evaluate_episodes = train_config.train.evaluate_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_value_estimator()
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer

    # choose Graph or Vanilla
    trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer,
                          env.human_num,
                          reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                          freeze_state_predictor=train_config.train.freeze_state_predictor,
                          detach_state_predictor=train_config.train.detach_state_predictor,
                          share_graph_model=policy_config.model_predictive_rl.share_graph_model)

    explorer = Explorer(env, agent, device, writer, memory, policy.gamma, target_policy=policy)

    logging.info('We use random-exploration methods to warm-up.')
    trainer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    agent.set_policy(policy)
    agent.print_info()
    env.set_agent(agent)
    trainer.set_learning_rate(rl_learning_rate)

    # fill the memory pool with some experience
    agent.policy.set_epsilon(1)
    explorer.run_k_episodes(k=warmup_episodes, phase='train', args=args, update_memory=True, plot_index=-1)  # 100
    logging.info('Warm-up finished!')
    logging.info('Experience set size: %d/%d\n', len(memory), memory.capacity)

    episode = 0
    best_val_reward = -1
    best_val_model = None

    while episode < num_episodes:
        # epsilon-greedy
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        agent.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(k=sample_episodes, phase='train', args=args, update_memory=True, plot_index=-1)

        explorer.log('train', episode)
        trainer.optimize_batch(num_batches, episode)
        logging.info(f"ep {episode} training is finished. epsilon={epsilon}\n")

        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            average_reward, _, _, _ ,_,_ = explorer.run_k_episodes(k=evaluate_episodes, phase='val', args=args,
                                                              plot_index=-1)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and average_reward > best_val_reward:
                logging.info("Best reward model has been changed.")
                best_val_reward = average_reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
            # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_k_episodes(k=1, phase='test', args=args, plot_index=episode)
                explorer.log('test', episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/infocom_benchmark/mp_separate_dp.py')
    parser.add_argument('--output_dir', type=str, default='logs/debug')  # output_xxxx
    parser.add_argument('--overwrite', default=False, action='store_true')

    parser.add_argument('--weights', type=str)
    parser.add_argument('--gpu_id', type=str, default='-1')
    parser.add_argument('--gpu', default=False, action='store_true')

    parser.add_argument('--debug', default=False, action='store_true')  # 开启debug模式
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=0)

    parser.add_argument('--vis_html', default=False, action='store_true')
    parser.add_argument('--plot_loop', default=False, action='store_true')
    parser.add_argument('--moving_line', default=False, action='store_true')
    sys_args = parser.parse_args()

    main(sys_args)

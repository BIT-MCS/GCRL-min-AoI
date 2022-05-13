import logging
import matplotlib.pyplot as plt
import torch


class Explorer(object):
    def __init__(self, env, robot, device, writer=None, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    # @profile
    def run_k_episodes(self, k, phase, args, plot_index, update_memory=False):
        self.robot.policy.set_phase(phase)
        cumulative_rewards = []
        average_return_list = []
        mean_aoi_list = []
        mean_energy_consumption_list = []
        collected_data_amount_list=[]
        update_human_coverage_list=[]

        for ep_i in range(k):
            state = self.env.unwrapped.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            returns = []
            while not done:
                action = self.robot.act(state, self.env.current_timestep)
                # print(self.env.start_timestamp+self.env.current_timestep*self.env.step_time,action)
                state, reward, done, info = self.env.unwrapped.step(action)  # 东西存在info里
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                if done:
                    mean_aoi_list.append(info["performance_info"]["mean_aoi"])
                    mean_energy_consumption_list.append(info["performance_info"]["mean_energy_consumption"])
                    collected_data_amount_list.append(info["performance_info"]["collected_data_amount"])
                    update_human_coverage_list.append(info["performance_info"]["human_coverage"])

            if update_memory:
                self.update_memory(states, actions, rewards)
                # if isinstance(info, ReachGoal) or isinstance(info, Collision):
                #     # only add positive(success) or negative(collision) experience in experience set
                #     self.update_memory(states, actions, rewards, imitation_learning)

            # calculate Bellman cumulative reward
            cumulative_rewards.append(sum([pow(self.gamma, t) * reward for t, reward in enumerate(rewards)]))
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t) * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_return_list.append(average(returns))

            if plot_index > 0:
                if args.vis_html:
                    self.env.render(mode='html', output_file=args.output_dir + f"/{phase}_page_{plot_index}.html",
                                    plot_loop=args.plot_loop,moving_line=args.moving_line)

        logging.info(f"cumulative_rewards:{average(cumulative_rewards)},  "
                     f"return:{average(average_return_list)},  "
                     f"mean_aoi: {average(mean_aoi_list)},  "
                     f"mean_energy_consumption: {average(mean_energy_consumption_list)}  "
                     f"collected_data_amount: {average(collected_data_amount_list)}  "
                     f"user_coverage: {average(update_human_coverage_list)}")


        if phase in ['val', 'test']:
            pass
            # total_time = sum(success_times + collision_times + timeout_times)
            # logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
            #              discomfort / total_time, average(min_dist))

        self.statistics = average(cumulative_rewards), average(average_return_list), average(mean_aoi_list), \
                          average(mean_energy_consumption_list), average(collected_data_amount_list), \
                          average(update_human_coverage_list)

        return self.statistics

    def update_memory(self, states, actions, rewards):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):
            reward = rewards[i]
            next_state = states[i + 1]
            if i == len(states) - 1:
                # terminal state
                value = reward
            else:
                value = 0

            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            self.memory.push((state[0], state[1], value, reward, next_state[0], next_state[1]))

    def log(self, tag_prefix, global_step):
        reward, avg_return, aoi, energy_consumption, collected_data_amount, ave_human_coverage = self.statistics
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)
        self.writer.add_scalar(tag_prefix + '/mean_human_aoi', aoi, global_step)
        self.writer.add_scalar(tag_prefix + '/energy_consumption (J)', energy_consumption, global_step)
        self.writer.add_scalar(tag_prefix + '/collected_data_amount (MB)', collected_data_amount, global_step)
        self.writer.add_scalar(tag_prefix + '/avg user coverage', ave_human_coverage, global_step)



def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

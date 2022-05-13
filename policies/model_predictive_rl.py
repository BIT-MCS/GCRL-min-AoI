import logging
import time
import numpy as np
import torch.multiprocessing as mp
from policies.base import Policy
from envs.model.mdp import build_action_space
from envs.model.utils import *
from configs.config import BaseEnvConfig
from method.state_predictor import StatePredictor
from method.graph_model import RGL
from method.value_estimator import ValueEstimator


class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
        self.trainable = True
        self.epsilon = None
        self.gamma = None
        self.action_space = None
        self.action_values = None
        self.share_graph_model = None
        self.value_estimator = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.robot_state_dim = None
        self.human_state_dim = None
        self.device = None

    def configure(self, config, human_df):
        self.gamma = config.rl.gamma
        self.robot_state_dim = config.model_predictive_rl.robot_state_dim
        self.human_state_dim = config.model_predictive_rl.human_state_dim
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.human_df = human_df
        self.tmp_config = BaseEnvConfig()

        if self.share_graph_model:  # perform worse than separated model
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, graph_model)
            self.state_predictor = StatePredictor(config, graph_model, self.device)
            self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
        else:
            graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = ValueEstimator(config, graph_model1)
            graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.state_predictor = StatePredictor(config, graph_model2, self.device)
            self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                          self.state_predictor.human_motion_predictor]

        if tmp_config.env.rollout_num == 1:
            for model in self.model:
                model.to(self.device)
        else:
            mp.set_start_method('spawn')
            for model in self.model:
                model.share_memory()
                model.to(self.device)

        logging.info('Planning depth: {}'.format(self.planning_depth))
        logging.info('Planning width: {}'.format(self.planning_width))
        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing d-step planning without action space clipping!')

        # env config
        self.human_num = self.tmp_config.env.human_num
        self.robot_num = self.tmp_config.env.robot_num
        self.num_timestep = self.tmp_config.env.num_timestep
        self.step_time = self.tmp_config.env.step_time
        self.start_timestamp = self.tmp_config.env.start_timestamp
        self.max_uav_energy = self.tmp_config.env.max_uav_energy

    def set_device(self, device):
        self.device = device

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_value_estimator(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
            else:
                return {
                    'graph_model1': self.value_estimator.graph_model.state_dict(),
                    'graph_model2': self.state_predictor.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
        else:
            return {
                'graph_model': self.value_estimator.graph_model.state_dict(),
                'value_network': self.value_estimator.value_network.state_dict()
            }

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            else:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
                self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
            self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def predict(self, state, current_timestep):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        # if self.reach_destination(state):
        #     return ActionXY(0, 0)
        if self.action_space is None:
            self.action_space = build_action_space()

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_value = float('-inf')
            max_traj = None

            if self.do_action_clip:  # True
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                if tmp_config.env.rollout_num == 1:
                    action_space_clipped = self.action_clip_single_process(state_tensor, self.action_space,
                                                                           self.planning_width,
                                                                           current_timestep)
                else:
                    action_space_clipped = self.action_clip_multi_processing(state_tensor, self.action_space,
                                                                             self.planning_width,
                                                                             current_timestep)
            else:
                action_space_clipped = self.action_space

            for action in action_space_clipped:  # action space剪枝
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                next_state = self.state_predictor(state_tensor, action)
                max_next_return, max_next_traj = self.V_planning(next_state, self.planning_depth, self.planning_width,
                                                                 current_timestep)
                reward_est = self.estimate_reward(state, action, current_timestep)
                value = reward_est + self.gamma * max_next_return
                if value > max_value:
                    max_value = value
                    max_action = action
                    max_traj = [(state_tensor, action, reward_est)] + max_next_traj

            # logging.info(f"clipped_action_space: {action_space_clipped}")
            # logging.info(f"max_action: {max_action}")
            if max_action is None:
                print(max_action)
                raise ValueError('Value network is not well trained.')

        self.last_state = state.to_tensor(device=self.device)

        return max_action

    def action_clip_single_process(self, state, action_space, width, current_timestep):
        values = []
        depth = 1
        # logging.info("start")
        for action in action_space:
            next_state_est = self.state_predictor(state, action)
            next_return, _ = self.V_planning(next_state_est, depth, width, current_timestep)
            reward_est = self.estimate_reward(state, action, current_timestep)
            value = reward_est + self.gamma * next_return
            values.append(value.item())
        # logging.info("end")
        max_indexes = np.argpartition(np.array(values), -width)[-width:]
        clipped_action_space = np.array([action_space[i] for i in max_indexes])

        # print(clipped_action_space)
        return clipped_action_space

    def action_value_estimate(self, current_dim, values, state, actions, current_timestep):
        for index, action in enumerate(actions):
            next_state_est = self.state_predictor(state, action)
            next_return = self.value_estimator(next_state_est)
            reward_est = self.estimate_reward(state, action, current_timestep)
            value = reward_est + self.gamma * next_return
            values[current_dim + index] = value.item()

    def any_process_alive(self, processes):
        for p in processes:
            if p.is_alive():
                return True
        return False

    def action_clip_multi_processing(self, state, action_space, width, current_timestep):
        # logging.info("start")
        values = torch.zeros([pow(9, self.tmp_config.env.robot_num), ], requires_grad=False)
        values.share_memory_()
        current_dim = 0
        processes = []

        for actions in np.array_split(action_space, tmp_config.env.rollout_num, axis=0):
            p = mp.Process(target=self.action_value_estimate,
                           args=(current_dim, values, [s.detach() for s in state], actions, current_timestep))
            current_dim += actions.shape[0]
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while True:
            if self.any_process_alive(processes):
                time.sleep(1)
            else:
                # print(values)
                max_indexes = torch.topk(values, width).indices
                clipped_action_space = np.array([action_space[i] for i in max_indexes])
                del values
                while True:
                    if self.any_process_alive(processes):
                        for p in processes:
                            p.close()
                    else:
                        break
                # logging.info("end")
                return clipped_action_space

    def V_planning(self, state, depth, width, current_timestep):  # 递归
        current_state_value = self.value_estimator(state)
        if depth == 1:
            return current_state_value, [(state, None, None)]

        if self.do_action_clip:
            if tmp_config.env.rollout_num == 1:
                action_space_clipped = self.action_clip_single_process(state, self.action_space, width,
                                                                       current_timestep)
            else:
                action_space_clipped = self.action_clip_multi_processing(state, self.action_space, width,
                                                                         current_timestep)
        else:
            action_space_clipped = self.action_space

        returns = []
        trajs = []

        for action in action_space_clipped:
            next_state_est = self.state_predictor(state, action)
            reward_est = self.estimate_reward(state, action, current_timestep)
            next_value, next_traj = self.V_planning(next_state_est, depth - 1, self.planning_width, current_timestep)
            return_value = current_state_value / depth + (depth - 1) / depth * (self.gamma * next_value + reward_est)

            returns.append(return_value.item())
            trajs.append([(state, action, reward_est)] + next_traj)

        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]

        return max_return, max_traj

    # TODO: 不太好改
    def estimate_reward(self, state, action, current_timestep):
        if isinstance(state, list) or isinstance(state, tuple):
            state = tensor_to_joint_state(state)
        human_states = state.human_states
        robot_states = state.robot_states
        current_human_aoi_list = np.zeros([self.human_num, ])
        next_human_aoi_list = np.zeros([self.human_num, ])
        current_uav_position = np.zeros([self.robot_num, 2])
        new_robot_position = np.zeros([self.robot_num, 2])
        current_robot_enenrgy_list = np.zeros([self.robot_num, ])
        next_robot_enenrgy_list = np.zeros([self.robot_num, ])
        current_enenrgy_consume = np.zeros([self.robot_num, ])
        num_updated_human = 0

        for robot_id, robot in enumerate(robot_states):
            new_robot_px = robot.px + action[robot_id][0]
            new_robot_py = robot.py + action[robot_id][1]
            is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
            is_collide = True if judge_collision(new_robot_px, new_robot_py, robot.px, robot.py) else False

            if is_stopping is True:
                consume_energy = consume_uav_energy(0, self.step_time)
            else:
                consume_energy = consume_uav_energy(self.step_time, 0)
            current_enenrgy_consume[robot_id] = consume_energy / tmp_config.env.max_uav_energy
            new_energy = robot.energy - consume_energy

            current_uav_position[robot_id][0] = robot.px
            current_uav_position[robot_id][1] = robot.py
            if is_collide:
                new_robot_position[robot_id][0] = robot.px
                new_robot_position[robot_id][1] = robot.py
            else:
                new_robot_position[robot_id][0] = new_robot_px
                new_robot_position[robot_id][1] = new_robot_py
            current_robot_enenrgy_list[robot_id] = robot.energy
            next_robot_enenrgy_list[robot_id] = new_energy

        selected_data, selected_next_data = get_human_position_list(current_timestep + 1, self.human_df)

        for human_id, human in enumerate(human_states):
            current_human_aoi_list[human_id] = human.aoi
            next_px, next_py, next_theta = get_human_position_from_list(current_timestep + 1, human_id, selected_data,
                                                                        selected_next_data)
            should_reset = judge_aoi_update([next_px, next_py], new_robot_position)
            if should_reset:
                next_human_aoi_list[human_id] = 1
                num_updated_human += 1
            else:
                next_human_aoi_list[human_id] = human.aoi + 1

        reward = np.mean(current_human_aoi_list - next_human_aoi_list) \
                 - tmp_config.env.energy_factor * np.sum(current_enenrgy_consume)

        return reward

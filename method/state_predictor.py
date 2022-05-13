import torch.nn as nn
import numpy as np
import torch
from method.base import mlp
from envs.model.utils import *
from configs.config import BaseEnvConfig


class StatePredictor(nn.Module):
    def __init__(self, config, graph_model, device):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.graph_model = graph_model
        self.human_motion_predictor = mlp(config.gcn.X_dim, config.model_predictive_rl.motion_predictor_dims)
        self.tmp_config = BaseEnvConfig()
        self.device = device

    def forward(self, state, action, detach=False):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        state_embedding = self.graph_model(state)
        if detach:
            state_embedding = state_embedding.detach()
        if action is None:
            # for training purpose
            next_robot_state = None
        else:
            next_robot_state = self.compute_next_state(state[0], action)
        # predict
        next_human_states = self.human_motion_predictor(state_embedding)[:, self.tmp_config.env.robot_num:,:]

        next_observation = [next_robot_state, next_human_states]
        return next_observation

    def compute_next_state(self, robot_states, action):
        robot_states_clone = robot_states.clone()
        robot_states_clone = tensor_to_robot_states(robot_states_clone)
        next_state_list = []
        for robot_id, robot_state in enumerate(robot_states_clone):
            new_robot_px = robot_state.px + action[robot_id][0]
            new_robot_py = robot_state.py + action[robot_id][1]
            is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
            is_collide = True if judge_collision(new_robot_px, new_robot_py, robot_state.px, robot_state.py) else False
            if is_stopping is True:
                new_energy = robot_state.energy - consume_uav_energy(0, self.tmp_config.env.step_time)
            else:
                new_energy = robot_state.energy - consume_uav_energy(self.tmp_config.env.step_time, 0)

            robot_theta = get_theta(0, 0, action[robot_id][0], action[robot_id][1])

            if is_collide:
                next_state_list.append([robot_state.px / self.tmp_config.env.nlon,
                                        robot_state.py / self.tmp_config.env.nlat,
                                        robot_theta / self.tmp_config.env.rotation_limit,
                                        new_energy / self.tmp_config.env.max_uav_energy])
            else:
                next_state_list.append([new_robot_px / self.tmp_config.env.nlon,
                                        new_robot_py / self.tmp_config.env.nlat,
                                        robot_theta / self.tmp_config.env.rotation_limit,
                                        new_energy / self.tmp_config.env.max_uav_energy])

        next_robot_states = torch.tensor(next_state_list, dtype=torch.float32)

        return next_robot_states.unsqueeze(0).to(self.device)
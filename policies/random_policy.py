import numpy as np
from policies.base import Policy
from envs.model.mdp import build_action_space


class RandomPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'RandomPolicy'
        self.action_space = None

    def configure(self, config,human_df):
        return

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def predict(self, state,current_timestep):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.action_space is None:
            self.action_space = build_action_space()

        max_action = self.action_space[np.random.choice(len(self.action_space))]

        return max_action

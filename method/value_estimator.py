import torch
import torch.nn as nn
from method.base import mlp
from configs.config import BaseEnvConfig


class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(config.gcn.X_dim, config.model_predictive_rl.value_network_dims)

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        tmp_config=BaseEnvConfig()
        robot_num=tmp_config.env.robot_num

        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        state_embedding = self.graph_model(state)[:, 0:robot_num, :]
        values = self.value_network(state_embedding)  # batch,robot num, 1
        value = values.mean(dim=1)
        return value

from configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'model_predictive_rl'

        self.model_predictive_rl = Config()
        self.model_predictive_rl.robot_state_dim = 4
        self.model_predictive_rl.human_state_dim = 4
        self.model_predictive_rl.planning_depth = 1  # 1 -> 2  shallow to dp
        self.model_predictive_rl.planning_width = 5  # 1 -> 5
        self.model_predictive_rl.do_action_clip = True  # False -> True
        self.model_predictive_rl.motion_predictor_dims = [32, 256, 256, self.model_predictive_rl.human_state_dim]
        self.model_predictive_rl.value_network_dims = [32, 256, 256, 1]
        self.model_predictive_rl.share_graph_model = False  # False!



class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        # trainer的小trick，暂时用不上
        self.train.freeze_state_predictor = False
        self.train.detach_state_predictor = False
        self.train.reduce_sp_update_frequency = False

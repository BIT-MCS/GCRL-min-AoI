from policies.model_predictive_rl import ModelPredictiveRL
from policies.random_policy import RandomPolicy

def none_policy():
    return None

policy_factory=dict()
policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['random_policy'] = RandomPolicy
policy_factory['none'] = none_policy




"""
Never Modify this file! Always copy the settings you want to change to your local file.
"""


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.num_timestep = 120  # 120x15=1800s=30min
    env.step_time = 15  # second per step
    env.max_uav_energy = 359640  # 359640 J <-- 359.64 kJ (4500mah, 22.2v) 大疆经纬
    env.rotation_limit = 360
    env.diameter_of_human_blockers = 0.5  # m
    env.h_rx = 1.3  # m, height of RX
    env.h_b = 1.7  # m, height of a human blocker
    env.velocity = 18
    env.frequence_band = 28  # GHz
    env.h_d = 120  # m, height of drone-BS
    env.alpha_nlos = 113.63
    env.beta_nlos = 1.16
    env.zeta_nlos = 2.58  # Frequency 28GHz, sub-urban. channel model
    env.alpha_los = 84.64
    env.beta_los = 1.55
    env.zeta_los = 0.12
    env.g_tx = 0  # dB
    env.g_rx = 5  # dB
    env.tallest_locs = None  # obstacle
    env.no_fly_zone = None  # obstacle
    env.start_timestamp = 1519894800
    env.end_timestamp = 1519896600
    env.energy_factor = 3  # TODO: energy factor in reward function
    env.robot_num = 2  # TODO: 多了要用多进程
    if env.robot_num > 15:
        env.rollout_num = env.robot_num * (env.robot_num - 1)  # 1 2 6 12 15
    else:
        env.rollout_num = 1

    # # TODO: purdue datasets
    env.lower_left = [-86.93, 40.4203]  # 经纬度
    env.upper_right = [-86.9103, 40.4313]
    env.nlon = 200
    env.nlat = 120
    env.human_num = 59
    env.dataset_dir = 'envs/crowd_sim/dataset/purdue/59 users.csv'
    env.sensing_range = 23.2  # unit  23.2
    env.one_uav_action_space = [[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30], [21, 21], [21, -21], [-21, 21],
                                [-21, -21]]
    env.max_x_distance = 1667  # m
    env.max_y_distance = 1222  # m
    env.density_of_human_blockers = 30000 / env.max_x_distance / env.max_y_distance  # block/m2

    # TODO: NCSU
    # env.lower_left = [-78.6988, 35.7651]
    # env.upper_right = [-78.6628, 35.7896]
    # env.nlon = 3600
    # env.nlat = 2450
    # env.human_num = 33
    # env.dataset_dir = 'envs/crowd_sim/dataset/NCSU/33 users.csv'
    # env.sensing_range = 220  # unit  220
    # env.one_uav_action_space = [[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210],
    #                             [-210, -210]]
    # env.max_x_distance = 3255.4913305859623
    # env.max_y_distance = 2718.3945272795013
    # env.density_of_human_blockers = 30000 / env.max_x_distance / env.max_y_distance  # block/m2

    # TODO: KAIST
    # env.lower_left = [127.3475, 36.3597]
    # env.upper_right = [127.3709, 36.3793]
    # env.nlon = 2340
    # env.nlat = 1960
    # env.human_num = 92
    # env.dataset_dir = 'envs/crowd_sim/dataset/KAIST/92 users.csv'
    # env.sensing_range = 220  # unit  220
    # env.one_uav_action_space = [[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210],
    #                             [-210, -210]]
    # env.max_x_distance = 2100.207579392558
    # env.max_y_distance = 2174.930950809533
    # env.density_of_human_blockers = 30000 / env.max_x_distance / env.max_y_distance  # block/m2

    # TODO: San Francisco datasets, SEE YOU NEXT TIME
    # env.lower_left = [-122.4620, 37.7441]
    # env.upper_right = [-122.3829, 37.8137]
    # env.nlon = 7910
    # env.nlat = 6960
    # env.human_num = 100
    # env.velocity = 18
    # env.dataset_dir = 'envs/crowd_sim/dataset/san/processed_train_half_100_data.csv'
    # env.start_timestamp = 1519894800
    # env.end_timestamp = 1519896600
    # env.sensing_range = 240  # unit  240
    # env.one_uav_action_space = [[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210],
    #                             [-210, -210]]
    # env.alpha_nlos = 66.25
    # env.beta_nlos = 3.3
    # env.zeta_nlos = 4.48  # Frequency 28GHz, high-rise building
    # env.alpha_los = 88.76
    # env.beta_los = 1.68
    # env.zeta_los = 2.47
    # env.max_x_distance = 6951  # m
    # env.max_y_distance = 7734  # m
    # env.density_of_human_blockers = 1057 / env.max_x_distance / env.max_y_distance  # block/m2
    # env.tallest_locs = [(37.7899, -122.3969), (37.7952, -122.4028), (37.7897, -122.3953), (37.7919, -122.4038),
    #                     (37.7925, -122.4005), (37.7904, -122.3961), (37.7858, -122.3921), (37.7878, -122.3942),
    #                     (37.7903, -122.3942), (37.7905, -122.3972), (37.7929, -122.3979), (37.7895, -122.4003),
    #                     (37.7952, -122.3961), (37.7945, -122.3997), (37.7898, -122.4018), (37.7933, -122.3945),
    #                     (37.7904, -122.4013), (37.7864, -122.3921), (37.7918, -122.3988), (37.7905, -122.3991),
    #                     (37.7887, -122.4026), (37.7911, -122.3981), (37.7861, -122.4025), (37.7891, -122.4033),
    #                     (37.7906, -122.403), (37.7853, -122.4109), (37.7916, -122.3958), (37.794, -122.3974),
    #                     (37.7885, -122.3986), (37.7863, -122.4013), (37.7926, -122.3989), (37.7912, -122.3971),
    #                     (37.7919, -122.3975), (37.7928, -122.4052), (37.7887, -122.3922), (37.7892, -122.3975),
    #                     (37.787, -122.3927), (37.7872, -122.392), (37.7872, -122.3953), (37.7932, -122.3972),
    #                     (37.7849, -122.4043), (37.7912, -122.4028), (37.787, -122.4), (37.7859, -122.3938),
    #                     (37.79, -122.3917), (37.7894, -122.3907), (37.7888, -122.3994), (37.7867, -122.4019),
    #                     (37.7912, -122.395), (37.7951, -122.3974), (37.7949, -122.3985), (37.7909, -122.3967),
    #                     (37.7893, -122.4008), (37.7919, -122.3945), (37.7904, -122.4024), (37.7939, -122.4004),
    #                     (37.7767, -122.4192), (37.7887, -122.3915), (37.7737, -122.4183)]
    # env.no_fly_zone = [[(4370.0, 3370.0), (4370.0, 3180.0), (4180.0, 3180.0), (4180.0, 3370.0)],
    #                    [(4470.0, 3020.0), (4470.0, 2880.0), (4280.0, 2880.0), (4280.0, 3020.0)],
    #                    [(5200.0, 4200.0), (5200.0, 4050.0), (5020.0, 4050.0), (5020.0, 4200.0)],
    #                    [(7100.0, 4450.0), (7100.0, 4090.0), (6580.0, 4090.0), (6580.0, 4450.0)],
    #                    [(7240.0, 4680.0), (7240.0, 4380.0), (6880.0, 4380.0), (6880.0, 4680.0)],
    #                    [(6300.0, 4360.0), (6300.0, 4000.0), (5660.0, 4000.0), (5660.0, 4360.0)],
    #                    [(6880.0, 5200.0), (6880.0, 4350.0), (6060.0, 4350.0), (6060.0, 5200.0)],
    #                    [(6130.0, 4960.0), (6130.0, 4360.0), (5540.0, 4360.0), (5540.0, 4960.0)],
    #                    [(6010.0, 5190.0), (6010.0, 5020.0), (5830.0, 5020.0), (5830.0, 5190.0)]]
    # env.g_tx = 10  # dB
    # env.g_rx = 5  # dB

    def __init__(self, debug=False):
        pass


class BasePolicyConfig(object):
    rl = Config()
    rl.gamma = 0.95

    gcn = Config()
    gcn.num_layer = 2
    gcn.X_dim = 32
    gcn.wr_dims = [64, gcn.X_dim]
    gcn.wh_dims = [64, gcn.X_dim]
    gcn.final_state_dim = gcn.X_dim
    gcn.similarity_function = 'embedded_gaussian'
    gcn.layerwise_graph = False
    gcn.skip_connection = True

    def __init__(self, debug=False):
        pass


class BaseTrainConfig(object):
    train = Config()
    train.rl_learning_rate = 0.001
    train.num_episodes = 500  # TODO:500
    train.warmup_episodes = 100  # TODO: 100, exploration
    train.evaluate_episodes = 1  # TODO: 10? 1?
    train.sample_episodes = 1

    # number of episodes sampled in one training episode
    train.target_update_interval = 30  # TODO:30
    train.evaluation_interval = 100  # TODO:100
    train.checkpoint_interval = 100
    train.num_batches = 100  # TODO:100, number of batches to train at the end of training episode

    # the memory pool can roughly store 2K episodes, total size = episodes * 50
    train.capacity = 50000
    train.epsilon_start = 0.5
    train.epsilon_end = 0.1
    train.epsilon_decay = 400

    trainer = Config()
    trainer.batch_size = 128
    trainer.optimizer = 'Adam'

    def __init__(self, debug=False):
        if debug:
            self.train.warmup_episodes = 1
            self.train.train_episodes = 10
            self.train.evaluation_interval = 5
            self.train.target_update_interval = 5


# r:meters, 2d distance
# threshold: dB
def try_sensing_range(r):
    import math
    config = BaseEnvConfig().env
    p_los = math.exp(
        -config.density_of_human_blockers * config.diameter_of_human_blockers * r * (config.h_b - config.h_rx) / (
                config.h_d - config.h_rx))
    p_nlos = 1 - p_los
    PL_los = config.alpha_los + config.beta_los * 10 * math.log10(
        math.sqrt(r * r + config.h_d * config.h_d)) + config.zeta_los
    PL_nlos = config.alpha_nlos + config.beta_nlos * 10 * math.log10(
        math.sqrt(r * r + config.h_d * config.h_d)) + config.zeta_nlos
    PL = p_los * PL_los + p_nlos * PL_nlos
    CL = PL - config.g_tx - config.g_rx
    print(p_los, p_nlos)
    print(CL)


# Maximum Coupling Loss (110dB is recommended)
# purdue:

# 123dB -> 560m -> 60.5 range
# 121dB -> 420m -> 45.4 range
# 119dB -> 300m -> 32.4 range
# 117dB -> 215m -> 23.2 range √
# 115dB -> 140m -> 15 range

# ncsu:
# 123dB -> 600m -> 600 range
# 121dB -> 435m -> 435 range
# 119dB -> 315m -> 315 range
# 117dB -> 220m -> 220 range √
# 115dB -> 145m -> 145 range

# kaist:
# 123dB -> 600m -> 600 range
# 121dB -> 435m -> 435 range
# 119dB -> 315m -> 315 range
# 117dB -> 220m -> 220 range √
# 115dB -> 145m -> 145 range

# san:
# 123dB -> 600m -> 600 range
# 121dB -> 450m -> 450 range
# 119dB -> 330m -> 330 range
# 117dB -> 240m -> 240 range √
# 115dB -> 165m -> 165 range


if __name__ == "__main__":
    try_sensing_range(220)

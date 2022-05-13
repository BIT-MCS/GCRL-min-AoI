import pandas as pd

import logging
import random
import gym
from envs.model.utils import *
from shapely.geometry import Point
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson, AntPath

from envs.model.mdp import HumanState, RobotState, JointState
import numpy as np
from configs.config import BaseEnvConfig
import os
import warnings


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.time_limit = None
        self.robots = None
        self.humans = None
        self.agent = None
        self.current_timestep = None
        self.phase = None

        self.config = BaseEnvConfig()

        self.human_num = self.config.env.human_num
        self.robot_num = self.config.env.robot_num
        self.num_timestep = self.config.env.num_timestep
        self.step_time = self.config.env.step_time
        self.start_timestamp = self.config.env.start_timestamp
        self.max_uav_energy = self.config.env.max_uav_energy

        # load_dataset
        self.nlon = self.config.env.nlon
        self.nlat = self.config.env.nlat
        self.lower_left = self.config.env.lower_left
        self.upper_right = self.config.env.upper_right
        self.human_df = pd.read_csv(self.config.env.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))
        # # 临时处理数据使用
        # sample_list=np.random.choice(self.human_num, size=[50,], replace=False)
        # sample_list=sample_list[np.argsort(sample_list)]
        # print(sample_list)
        # self.human_df= self.human_df[self.human_df["id"].isin(sample_list)]
        # for i,human_id in enumerate(sample_list):
        #     mask=(self.human_df["id"]==human_id)
        #     self.human_df.loc[mask,"id"]=i
        # self.human_df=self.human_df.sort_values(by=["id","timestamp"],ascending=[True,True])
        # print(self.human_df.head())
        # self.human_df.to_csv("50 users-5.csv",index=False)
        # exit(0)

        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # s表示时间戳转换
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy
        logging.info('human number: {}'.format(self.human_num))
        logging.info('Robot number: {}'.format(self.robot_num))

        # for debug
        self.current_human_aoi_list = np.ones([self.human_num, ])
        self.mean_aoi_timelist = np.ones([self.config.env.num_timestep + 1, ])
        self.robot_energy_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_x_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_y_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.update_human_timelist = np.zeros([self.config.env.num_timestep, ])
        self.data_collection = 0

    def set_agent(self, agent):
        self.agent = agent

    def generate_human(self, human_id, selected_data, selected_next_data):
        human = Human(human_id, self.config)
        px, py, theta = get_human_position_from_list(self.current_timestep, human_id, selected_data, selected_next_data)
        human.set(px, py, theta, 1)  # human有aoi
        return human

    def generate_robot(self, robot_id):
        robot = Robot(robot_id, self.config)
        robot.set(self.nlon / 2, self.nlat / 2, 0, self.max_uav_energy)  # robot有energy
        return robot

    def sync_human_df(self, human_id, current_timestep, aoi):
        current_timestamp = self.start_timestamp + current_timestep * self.step_time
        current_index = self.human_df[
            (self.human_df.id == human_id) & (self.human_df.timestamp == current_timestamp)].index
        # self.human_df.loc[current_index, "aoi"] = aoi   # slower
        self.human_df.iat[current_index.values[0], 9] = aoi

    def reset(self, phase='test', test_case=None):
        # 暂时先把随机种子定死。原始代码里存在有case size迭代随机种子的操作，这是好操作。
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        base_seed = {'train': 0, 'val': 0, 'test': 0}
        np.random.seed(base_seed[phase])
        random.seed(base_seed[phase])

        self.current_timestep = 0

        # 生成human
        self.humans = []
        selected_data, selected_next_data = get_human_position_list(self.current_timestep, self.human_df)
        for human_id in range(self.human_num):
            self.humans.append(self.generate_human(human_id, selected_data, selected_next_data))
            self.sync_human_df(human_id, self.current_timestep, 1)

        # 生成robot
        self.robots = []
        for robot_id in range(self.robot_num):
            self.robots.append(self.generate_robot(robot_id))

        self.current_human_aoi_list = np.ones([self.human_num, ])
        self.mean_aoi_timelist = np.ones([self.config.env.num_timestep + 1, ])
        self.mean_aoi_timelist[self.current_timestep] = np.mean(self.current_human_aoi_list)
        self.robot_energy_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_energy_timelist[self.current_timestep, :] = self.max_uav_energy
        self.robot_x_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_x_timelist[self.current_timestep, :] = self.nlon / 2
        self.robot_y_timelist = np.zeros([self.config.env.num_timestep + 1, self.robot_num])
        self.robot_y_timelist[self.current_timestep, :] = self.nlat / 2
        self.update_human_timelist = np.zeros([self.config.env.num_timestep, ])
        self.data_collection = 0

        # full和observaeble的概念可以去掉了。
        self.plot_states = list()
        self.robot_actions = list()
        self.rewards = list()
        self.action_values = list()
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])
        state = JointState([robot.get_obs() for robot in self.robots], [human.get_obs() for human in self.humans])

        return state

    def step(self, action):
        new_robot_position = np.zeros([self.robot_num, 2])
        current_enenrgy_consume = np.zeros([self.robot_num, ])

        num_updated_human = 0

        for robot_id, robot in enumerate(self.robots):
            new_robot_px = robot.px + action[robot_id][0]
            new_robot_py = robot.py + action[robot_id][1]
            robot_theta = get_theta(0, 0, action[robot_id][0], action[robot_id][1])
            # print(action[robot_id], robot_theta)
            is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
            is_collide = True if judge_collision(new_robot_px, new_robot_py, robot.px, robot.py) else False

            if is_stopping is True:
                consume_energy = consume_uav_energy(0, self.step_time)
            else:
                consume_energy = consume_uav_energy(self.step_time, 0)
            current_enenrgy_consume[robot_id] = consume_energy / self.config.env.max_uav_energy
            new_energy = robot.energy - consume_energy
            self.robot_energy_timelist[self.current_timestep + 1][robot_id] = new_energy

            if is_collide is True:
                new_robot_position[robot_id][0] = robot.px
                new_robot_position[robot_id][1] = robot.py
                self.robot_x_timelist[self.current_timestep + 1][robot_id] = robot.px
                self.robot_y_timelist[self.current_timestep + 1][robot_id] = robot.py
                robot.set(robot.px, robot.py, robot_theta, energy=new_energy)
            else:
                new_robot_position[robot_id][0] = new_robot_px
                new_robot_position[robot_id][1] = new_robot_py
                self.robot_x_timelist[self.current_timestep + 1][robot_id] = new_robot_px
                self.robot_y_timelist[self.current_timestep + 1][robot_id] = new_robot_py
                robot.set(new_robot_px, new_robot_py, robot_theta, energy=new_energy)

        selected_data, selected_next_data = get_human_position_list(self.current_timestep + 1, self.human_df)
        delta_human_aoi_list = np.zeros_like(self.current_human_aoi_list)
        for human_id, human in enumerate(self.humans):
            next_px, next_py, next_theta = get_human_position_from_list(self.current_timestep + 1, human_id,
                                                                        selected_data, selected_next_data)
            should_reset = judge_aoi_update([next_px, next_py], new_robot_position)
            if should_reset:
                if human.aoi > 1:
                    delta_human_aoi_list[human_id] = human.aoi
                else:
                    delta_human_aoi_list[human_id] = 1

                human.set(next_px, next_py, next_theta, aoi=1)
                num_updated_human += 1
            else:
                delta_human_aoi_list[human_id] = 0
                new_aoi = human.aoi + 1
                human.set(next_px, next_py, next_theta, aoi=new_aoi)

            self.current_human_aoi_list[human_id] = human.aoi
            self.sync_human_df(human_id, self.current_timestep + 1, human.aoi)

        self.mean_aoi_timelist[self.current_timestep + 1] = np.mean(self.current_human_aoi_list)
        self.update_human_timelist[self.current_timestep] = num_updated_human
        delta_sum_aoi = np.sum(delta_human_aoi_list)
        self.data_collection += (delta_sum_aoi * 0.3)  # Mb, 0.02M/s per person

        # TODO: need to be well-defined
        reward = self.mean_aoi_timelist[self.current_timestep] - self.mean_aoi_timelist[self.current_timestep + 1] \
                 - self.config.env.energy_factor * np.sum(current_enenrgy_consume)

        if hasattr(self.agent.policy, 'action_values'):
            self.action_values.append(self.agent.policy.action_values)
        self.robot_actions.append(action)
        self.rewards.append(reward)
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])

        next_state = JointState([robot.get_obs() for robot in self.robots],
                                [human.get_obs() for human in self.humans])

        self.current_timestep += 1
        if self.current_timestep >= self.num_timestep:
            done = True
        else:
            done = False
        info = {"performance_info": {
            "mean_aoi": self.mean_aoi_timelist[self.current_timestep],
            "mean_energy_consumption": 1.0 - (
                        np.mean(self.robot_energy_timelist[self.current_timestep]) / self.max_uav_energy),
            "collected_data_amount": self.data_collection/(self.num_timestep*self.human_num*0.3),
            "human_coverage": np.mean(self.update_human_timelist) / self.human_num
        },
        }

        return next_state, reward, done, info

    def render(self, mode='traj', output_file=None, plot_loop=False, moving_line=False):
        # -------------------------------------------------------------------
        if mode == 'html':
            import geopandas as gpd
            import movingpandas as mpd
            from movingpandas.geometry_utils import measure_distance_geodesic
            max_distance_x = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                       Point(self.upper_right[0], self.lower_left[1]))
            max_distance_y = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                       Point(self.lower_left[0], self.upper_right[1]))

            mixed_df = self.human_df.copy()

            # 可将机器人traj，可以载入到human的dataframe中，id从-1开始递减
            for i in range(self.robot_num):
                x_list = self.robot_x_timelist[:, i]
                y_list = self.robot_y_timelist[:, i]
                id_list = np.ones_like(x_list) * (-i - 1)
                aoi_list = np.ones_like(x_list) * (-1)
                energy_list = self.robot_energy_timelist[:, i]
                timestamp_list = [self.start_timestamp + i * self.step_time for i in range(self.num_timestep + 1)]
                x_distance_list = x_list * max_distance_x / self.nlon + max_distance_x / self.nlon / 2
                y_distance_list = y_list * max_distance_y / self.nlat + max_distance_y / self.nlat / 2
                max_longitude = abs(self.lower_left[0] - self.upper_right[0])
                max_latitude = abs(self.lower_left[1] - self.upper_right[1])
                longitude_list = x_list * max_longitude / self.nlon + max_longitude / self.nlon / 2 + self.lower_left[0]
                latitude_list = y_list * max_latitude / self.nlat + max_latitude / self.nlat / 2 + self.lower_left[1]

                data = {"id": id_list, "longitude": longitude_list, "latitude": latitude_list,
                        "x": x_list, "y": y_list, "x_distance": x_distance_list, "y_distance": y_distance_list,
                        "timestamp": timestamp_list, "aoi": aoi_list, "energy": energy_list}
                robot_df = pd.DataFrame(data)
                robot_df['t'] = pd.to_datetime(robot_df['timestamp'], unit='s')  # s表示时间戳转换
                mixed_df = mixed_df.append(robot_df)

            # ------------------------------------------------------------------------------------
            # 建立moving pandas轨迹，也可以选择调用高级API继续清洗轨迹。
            mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude),
                                         crs=4326)
            mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)  # tz=time zone, 以本地时间为准
            mixed_gdf = mixed_gdf.sort_values(by=["id", "t"], ascending=[True, True])
            trajs = mpd.TrajectoryCollection(mixed_gdf, 'id')
            # trajs = mpd.MinTimeDeltaGeneralizer(trajs).generalize(tolerance=timedelta(minutes=1))
            # for index, traj in enumerate(trajs.trajectories):
            #     print(f"id: {trajs.trajectories[index].df['id'][0]}"
            #           + f"  size:{trajs.trajectories[index].size()}")

            start_point = trajs.trajectories[0].get_start_location()

            # 经纬度反向
            m = folium.Map(location=[start_point.y, start_point.x], tiles="cartodbpositron", zoom_start=14, max_zoom=24)

            m.add_child(folium.LatLngPopup())
            minimap = folium.plugins.MiniMap()
            m.add_child(minimap)
            folium.TileLayer('Stamen Terrain').add_to(m)
            folium.TileLayer('Stamen Toner').add_to(m)
            folium.TileLayer('cartodbpositron').add_to(m)
            folium.TileLayer('OpenStreetMap').add_to(m)

            # 锁定范围
            grid_geo_json = get_border(self.upper_right, self.lower_left)
            color = "red"
            border = folium.GeoJson(grid_geo_json,
                                    style_function=lambda feature, color=color: {
                                        'fillColor': color,
                                        'color': "black",
                                        'weight': 2,
                                        'dashArray': '5,5',
                                        'fillOpacity': 0,
                                    })
            m.add_child(border)

            for index, traj in enumerate(trajs.trajectories):
                name = f"UAV {index}" if index < self.robot_num else f"Human {traj.df['id'][0]}"  # select human
                # name = f"UAV {index}" if index < robot_num else f"Human {index - robot_num}"
                randr = lambda: np.random.randint(0, 255)
                color = '#%02X%02X%02X' % (randr(), randr(), randr())  # black

                # point
                features = traj_to_timestamped_geojson(index, traj, self.robot_num, color)
                TimestampedGeoJson(
                    {
                        "type": "FeatureCollection",
                        "features": features,
                    },
                    period="PT15S",
                    add_last_point=True,
                    transition_time=5,
                    loop=plot_loop,
                ).add_to(m)  # sub_map

                # line
                if index < self.robot_num:
                    geo_col = traj.to_point_gdf().geometry
                    xy = [[y, x] for x, y in zip(geo_col.x, geo_col.y)]
                    f1 = folium.FeatureGroup(name)
                    if moving_line:
                        AntPath(locations=xy, color=color, weight=4, opacity=0.7, dash_array=[100, 20],
                                delay=1000).add_to(f1)
                    else:
                        folium.PolyLine(locations=xy, color=color, weight=4, opacity=0.7).add_to(f1)
                    f1.add_to(m)

            folium.LayerControl().add_to(m)

            if self.config.env.tallest_locs is not None:
                # 绘制正方形
                for tallest_loc in self.config.env.tallest_locs:
                    # folium.Rectangle(
                    #     bounds=[(tallest_loc[0] + 0.00025, tallest_loc[1] + 0.0003),
                    #             (tallest_loc[0] - 0.00025, tallest_loc[1] - 0.0003)],  # 解决经纬度在地图上的尺度不一致
                    #     color="black",
                    #     fill=True,
                    # ).add_to(m)
                    icon_square = folium.plugins.BeautifyIcon(
                        icon_shape='rectangle-dot',
                        border_color='red',
                        border_width=8,
                    )
                    folium.Marker(location=[tallest_loc[0], tallest_loc[1]],
                                  popup=folium.Popup(html=f'<p>raw coord: ({tallest_loc[1]},{tallest_loc[0]})</p>'),
                                  tooltip='High-rise building',
                                  icon=icon_square).add_to(m)

            m.get_root().render()
            m.get_root().save(output_file)
            logging.info(f"{output_file} saved!")
        # -----------------------------------------------------------------------------
        elif mode == 'traj':
            pass
        else:
            raise NotImplementedError

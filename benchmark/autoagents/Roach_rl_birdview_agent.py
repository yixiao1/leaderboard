#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
import os
import math
from queue import Queue
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy
from importlib import import_module

from benchmark.utils.route_manipulation import downsample_route
from benchmark.envs.sensor_interface import SensorInterface

from dataloaders.transforms import encode_directions_4, encode_directions_6,inverse_normalize, decode_directions_4
from benchmark.utils.waypointer import Waypointer
from omegaconf import OmegaConf
from network.models.architectures.Roach_rl_birdview.birdview.chauffeurnet import ObsManager
import network.models.architectures.Roach_rl_birdview.utils.transforms as trans_utils
from network.models.architectures.Roach_rl_birdview.utils.traffic_light import TrafficLightHandler

def checkpoint_parse_configuration_file(filename):

    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

def get_entry_point():
    return 'Roach_rl_birdview_agent'

class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

class Roach_rl_birdview_agent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, save_driving_vision, save_driving_measurement, save_to_hdf5):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.inputs_buffer=Queue()
        self.waypointer = None
        
        self.vision_save_path = save_driving_vision
        self.measurement_save_path = save_driving_measurement
        self.save_to_hdf5 = save_to_hdf5
        if self.save_to_hdf5:
            hdf5_dir = '/'.join(self.vision_save_path.split('/')[:-2])
            hdf5_name = '_'.join(self.vision_save_path.split('/')[-2:])
            self.hdf5_save_path = os.path.join(hdf5_dir, hdf5_name)

        # agent's initialization
        self.setup_model(path_to_conf_file)

        self.hf = None
        self.cmap_2 = plt.get_cmap('jet')
        self.datapoint_count = 0

        self.register_actor=[]


    def setup_model(self, path_to_conf_file):
        exp_dir = os.path.join('/', os.path.join(*path_to_conf_file.split('/')[:-1]))
        yaml_conf, checkpoint_number, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        cfg = OmegaConf.load(os.path.join(exp_dir, yaml_conf))

        self._ckpt = os.path.join(exp_dir, 'checkpoints', str(checkpoint_number) + '.pth')

        cfg = OmegaConf.to_container(cfg)

        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']

        # prepare policy
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']
        print(f'Loading checkpoint: {self._ckpt}')
        self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
        self._policy = self._policy.eval()

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

        self._obs_managers= ObsManager(cfg['obs_configs']['birdview'])

    def set_world(self, world):
        self.world=world
        self.map=self.world.get_map()

        TrafficLightHandler.reset(self.world)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 10000000)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in
                                         ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._route_plan = global_plan_world_coord
        self.waypointer = Waypointer(self._global_plan, self._global_plan[0][0], self.world)

    def set_ego_vehicle(self, ego_vehicle):
        self._ego_vehicle=ego_vehicle
        self._obs_managers.attach_ego_vehicle(self._ego_vehicle, self._route_plan)

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 900, 'height': 256, 'fov': 100, 'id': 'rgb_central'},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 7.0, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 450, 'height': 450, 'fov': 150, 'id': 'rgb_ontop'},

            {'type': 'sensor.other.gnss', 'id': 'GPS'},

            {'type': 'sensor.other.imu', 'id': 'IMU'},

            {'type': 'sensor.speedometer', 'id': 'SPEED'},

            {'type': 'sensor.can_bus', 'id': 'can_bus'}
        ]

        return sensors
    

    def __call__(self, timestamp):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        print('    Timestamp frame = ',timestamp.frame)

        for key, values in input_data.items():
            if values[0] != timestamp.frame:
                raise RuntimeError(' The frame number of sensor data does not match the timestamp frame:', key)

        input_data = self.adding_roach_data(input_data)
        
        inputs = [input_data]

        control = self.run_step(inputs)
        control.manual_gear_shift = False

        # We pop the first frame of the input buffer and stack the current frame to the end
        self.inputs_buffer.get()
        self.inputs_buffer.put(input_data)

        return control

    def calculate_velocity(self,actor):
        """
        Method to calculate the velocity of a actor
        """
        velocity_squared = actor.get_velocity().x ** 2
        velocity_squared += actor.get_velocity().y ** 2
        return math.sqrt(velocity_squared)

    def run_step(self, inputs_data):
        """
        Execute one step of navigation.
        :return: control
        """
        input_data = copy.deepcopy(inputs_data)

        policy_input = self._wrapper_class.process_obs(input_data[-1], self._wrapper_kwargs['input_states'], train=False)

        cmd = self.process_command(inputs_data[-1]['GPS'][1], inputs_data[-1]['IMU'][1])[1]

        actions, _, _, _, _, _ = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)

        """
        ## This part is for collecting curated scenario data
        ego_location = self._ego_vehicle.get_transform().location
        ego_wp = self.map.get_waypoint(ego_location)


        history_queue = self._obs_managers._history_queue
        # get the latest candidate obstacles in history
        vehicles, walkers, tl_green, tl_yellow, tl_red, _ = history_queue[-1]
        objects_within_20m_radius = []
        if vehicles+walkers:
            for actor_transform, _,_, actor in vehicles+walkers:
                vec_ego_actor_in_global = actor_transform.location - ego_location
                compass = 0.0 if np.isnan(inputs_data[-1]['IMU'][1][-1]) else inputs_data[-1]['IMU'][1][-1]
                ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
                loc_in_ev = self.waypointer.vec_global_to_ref(vec_ego_actor_in_global, ref_rot_in_global)
                if loc_in_ev.x < 0.0 or abs(loc_in_ev.y) > 10.0:
                    continue
                dist_ego_actor = math.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2)
                if dist_ego_actor < 20.0:
                    objects_within_20m_radius.append([actor, loc_in_ev])

        if objects_within_20m_radius:
            for actor, loc_in_ev in objects_within_20m_radius:
                actor_transform = actor.get_transform()
                actor_velocity = self.calculate_velocity(actor)
                if (loc_in_ev.y >= 0.0 and int(actor_transform.rotation.yaw - ego_wp.transform.rotation.yaw) % 360.0 in range(220, 320)) \
                        or (loc_in_ev.y < 0.0 and int(actor_transform.rotation.yaw - ego_wp.transform.rotation.yaw) % 360.0 in range(40, 140)):
                    #print('1')
                    #print('velocity', actor_velocity)
                    #print('loc_in_ev.y', loc_in_ev.y)

                    th1 = 1.7 + np.random.uniform(-0.1, 0.0)
                    th2 = 2.0 + np.random.uniform(-0.2, 0.0)
                    th3 = 0.8 + np.random.uniform(-0.05, 0.0)
                    th4 = 1.2 + np.random.uniform(-0.1, 0.00)

                    th5 = 1.2 + np.random.uniform(0.0, 0.1)
                    th6 = 1.0 + np.random.uniform(0.0, 0.1)
                    if -ego_wp.lane_width * th2 < loc_in_ev.y< -ego_wp.lane_width * th1 or ego_wp.lane_width * th3 < loc_in_ev.y< ego_wp.lane_width * th4:
                        if th6 <=actor_velocity < th5:
                            control = self.takeout_control(0.3+np.random.uniform(-0.1, 0.1))
                        elif actor_velocity >= th5:
                            control = self.takeout_control(0.5+np.random.uniform(-0.1, 0.1))

                    elif -ego_wp.lane_width * th1 < loc_in_ev.y < ego_wp.lane_width * th3:
                        if th6 <=actor_velocity < th4:
                            control = self.takeout_control(0.8+np.random.uniform(-0.1, 0.1))
                        elif actor_velocity >= th4:
                            control = self.takeout_control(1.0)
                elif (loc_in_ev.y >= 0.0 and int(actor_transform.rotation.yaw - ego_wp.transform.rotation.yaw) % 360.0 in range(40, 140)) \
                        or (loc_in_ev.y < 0.0 and int(actor_transform.rotation.yaw - ego_wp.transform.rotation.yaw) % 360.0 in range(220, 320)):

                    #print('2')
                    #print('velocity', actor_velocity)
                    #print('loc_in_ev.y', loc_in_ev.y)

                    if -ego_wp.lane_width * (0.8+ np.random.uniform(-0.1, 0.1)) < loc_in_ev.y < ego_wp.lane_width * (0.8+ np.random.uniform(-0.1, 0.05)):
                            control = self.takeout_control(1.0)

        """

        steer = control.steer
        throttle = control.throttle
        brake = control.brake

        if self.save_to_hdf5:
            if not os.path.exists('/'.join(self.hdf5_save_path.split('/')[:-1])):
                os.makedirs('/'.join(self.hdf5_save_path.split('/')[:-1]))
            if not self.hf:
                self.hf = h5py.File(self.hdf5_save_path, 'w')
            group_frame = self.hf.create_group(f"frame_{self.datapoint_count}")

        if self.vision_save_path:
            last_input = input_data[-1]['rgb_central'][1]
            last_input = Image.fromarray(last_input)

            #"""
            last_input_ontop = Image.fromarray(inputs_data[-1]['rgb_ontop'][1])

            if float(cmd) == 1.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '4_directions', 'turn_left.png'))

            elif float(cmd) == 2.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '4_directions', 'turn_right.png'))

            elif float(cmd) == 3.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '4_directions', 'go_straight.png'))

            elif float(cmd) == 4.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '4_directions', 'follow_lane.png'))

            else:
                raise RuntimeError()

            command_sign = command_sign.resize((260, 80))

            mat = Image.new('RGB', (
                last_input_ontop.width + last_input.width, max(last_input_ontop.height, last_input.height)),
                            (0, 0, 0))
            mat.paste(command_sign, (last_input_ontop.width + 450 , last_input.height + 20))

            mat.paste(last_input_ontop, (0, 0))
            mat.paste(last_input, (last_input_ontop.width, 0))
            birdview = Image.fromarray(input_data[-1]['birdview']['rendered']).resize(((last_input_ontop.height-last_input.height),
                                                                  (last_input_ontop.height-last_input.height)))
            mat.paste(birdview, (last_input_ontop.width, last_input.height))

            draw_mat = ImageDraw.Draw(mat)
            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 20)
            draw_mat.text((last_input_ontop.width + 240, last_input_ontop.height - 30),
                          str("Steer " + "%.3f" % steer), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 410, last_input_ontop.height - 30),
                          str("Throttle " + "%.3f" % throttle), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 580, last_input_ontop.height - 30),
                          str("Brake " + "%.3f" % brake), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 740, last_input_ontop.height - 30),
                              str("Speed " + "%.3f" % inputs_data[-1]['SPEED'][1]['speed']), fill=(255, 255, 255), font=font)
            #mat = mat.resize((650, 225))

            if self.save_to_hdf5:
                img_arr = np.array(mat)
                if img_arr.size > 2000:
                    group_frame.create_dataset('footage', data=img_arr, compression="gzip", compression_opts=4)
                else:
                    group_frame.create_dataset('footage', data=img_arr)

            else:
                if not os.path.exists(self.vision_save_path):
                    os.makedirs(self.vision_save_path)

                if not os.path.exists(os.path.join(self.vision_save_path,'check')):
                    os.makedirs(os.path.join(self.vision_save_path,'check'))
                mat.save(os.path.join(self.vision_save_path, 'check', str(self.datapoint_count).zfill(6) + '.jpg'))
                image = last_input.resize((600, 170))
                image.save(os.path.join(self.vision_save_path, 'rgb_central' + '%06d.png' % self.datapoint_count))

        if self.measurement_save_path:
            # we record the driving measurement data
            data = inputs_data[-1]['can_bus'][1]
            data.update({'steer': np.nan_to_num(control.steer)})
            data.update({'throttle': np.nan_to_num(control.throttle)})
            data.update({'brake': np.nan_to_num(control.brake)})
            data.update({'hand_brake': control.hand_brake})
            data.update({'reverse': control.reverse})
            data.update({'speed': inputs_data[-1]['SPEED'][1]['speed']})
            data.update({'direction': float(cmd)})

            if throttle == 0.0:
                if brake <= 1.0:
                    data['acceleration'] = -1 * brake
                else:
                    raise RuntimeError
            elif brake == 0.0:
                if throttle <= 1.0:
                    data['acceleration'] = throttle
                else:
                    raise RuntimeError
            else:
                raise RuntimeError

            if self.save_to_hdf5:
                for key, value in data.items():
                    if not isinstance(value, np.ndarray):
                        if not isinstance(value, list):
                            value = np.array([value])
                        else:
                            value = np.array(value)
                    group_frame.create_dataset(key, data=value)

            else:
                if not os.path.exists(self.measurement_save_path):
                    os.makedirs(self.measurement_save_path)
                with open(
                        os.path.join(self.measurement_save_path,
                                     'can_bus' + str(self.datapoint_count).zfill(6) + '.json'),
                        'w') as fo:
                    jsonObj = {}
                    jsonObj.update(data)
                    fo.seek(0)
                    fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
                    fo.close()

        self.datapoint_count+=1

        return control

    def adding_roach_data(self, input_dict):
        obs_dict = self._obs_managers.get_observation()
        input_dict.update({'birdview': obs_dict})

        control = self._ego_vehicle.get_control()
        speed_limit = self._ego_vehicle.get_speed_limit() / 3.6 * 0.8
        control_obs = {
            'throttle': np.array([control.throttle], dtype=np.float32),
            'steer': np.array([control.steer], dtype=np.float32),
            'brake': np.array([control.brake], dtype=np.float32),
            'gear': np.array([control.gear], dtype=np.float32),
            'speed_limit': np.array([speed_limit], dtype=np.float32),
        }

        ev_transform = self._ego_vehicle.get_transform()
        acc_w = self._ego_vehicle.get_acceleration()
        vel_w = self._ego_vehicle.get_velocity()
        ang_w = self._ego_vehicle.get_angular_velocity()

        acc_ev = trans_utils.vec_global_to_ref(acc_w, ev_transform.rotation)
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)

        velocity_obs = {
            'acc_xy': np.array([acc_ev.x, acc_ev.y], dtype=np.float32),
            'vel_xy': np.array([vel_ev.x, vel_ev.y], dtype=np.float32),
            'vel_ang_z': np.array([ang_w.z], dtype=np.float32)
        }

        input_dict.update({'control': control_obs})
        input_dict.update({'velocity': velocity_obs})

        return input_dict

    def stopping_and_wait(self):
        """
        The ego stops and waits until the input buffer is full
        :return:  control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

    def takeout_control(self, brake=None):
        """
        The ego stops and waits until the input buffer is full
        :return:  control
        """
        print('  Dangerous!! Taking out control!!')
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = brake
        control.hand_brake = False
        return control


    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        self._model = None
        self.checkpoint=None
        self.world=None
        self.map=None

        self.reset()

    def reset(self):
        self._global_plan = None
        self._global_plan_world_coord = None

        self.sensor_interface = None
        self.inputs_buffer = None
        self.waypointer = None
        self.vision_save_path = None
        self.datapoint_count = 0


    def process_command(self, gps, imu):
        _, _, cmd = self.waypointer.tick(gps, imu)

        return encode_directions_4(cmd.value), cmd.value

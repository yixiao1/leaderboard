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
import torch
from srunner.scenariomanager.timer import GameTime
from queue import Queue
import json
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy
from importlib import import_module

from benchmark.utils.route_manipulation import downsample_route
from benchmark.envs.sensor_interface import SensorInterface

from dataloaders.transforms import encode_directions_4, encode_directions_6,inverse_normalize
from benchmark.utils.waypointer import Waypointer
from benchmark.envs.data_writer import Writer
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

    def __init__(self, path_to_conf_file, save_sensor=None, save_attention=None):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.inputs_buffer=Queue()
        self.waypointer = None
        self.attention_save_path = None

        # agent's initialization
        self.setup(path_to_conf_file)
        self.wallclock_t0 = None

        if save_sensor is not None:
            self.writer = Writer(save_sensor)
        else:
            self.writer = None

        if save_attention:
            self.cmap_2 = plt.get_cmap('jet')
            self.attention_save_path = save_attention
            self.att_count = 0

    def setup(self, path_to_conf_file):
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

    def run_step(self, inputs_data):
        """
        Execute one step of navigation.
        :return: control
        """

        input_data = copy.deepcopy(inputs_data)

        policy_input = self._wrapper_class.process_obs(input_data[-1], self._wrapper_kwargs['input_states'], train=False)

        actions, _, _, _, _, _ = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)

        steer = control.steer
        throttle = control.throttle
        brake = control.brake

        if self.attention_save_path:
            if not os.path.exists(self.attention_save_path):
                os.makedirs(self.attention_save_path)
            last_input = input_data[-1]['rgb_central'][1]
            last_input = Image.fromarray(last_input)

            #att = np.delete(self.cmap_2(np.abs(att_backbone_layers[-1][-1][0].cpu().data.numpy()).mean(0) / np.abs(
            #    att_backbone_layers[-1][-1][0].cpu().data.numpy()).mean(0).max()), 3, 2)
            #att = np.array(Image.fromarray((att * 255).astype(np.uint8)).resize(
            #    (g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])))
            #last_att = Image.fromarray(att)
            #blend_im = Image.blend(last_input, last_att, 0.7)
            last_input_ontop = Image.fromarray(inputs_data[-1]['rgb_ontop'][1])

            cmd = self.process_command(inputs_data[-1]['GPS'][1], inputs_data[-1]['IMU'][1])[1]
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

            """
            elif float(cmd) == 5.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'change_left.png'))

            elif float(cmd) == 6.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'change_right.png'))
            """

            command_sign = command_sign.resize((390, 120))

            mat = Image.new('RGB', (
                last_input_ontop.width + last_input.width, max(last_input_ontop.height, last_input.height)),
                            (0, 0, 0))
            mat.paste(command_sign, (last_input_ontop.width + 255, last_input.height + 20))

            mat.paste(last_input_ontop, (0, 0))
            mat.paste(last_input, (last_input_ontop.width, 0))
            #mat.paste(blend_im, (last_input_ontop.width, last_input.height))

            draw_mat = ImageDraw.Draw(mat)
            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 20)
            draw_mat.text((last_input_ontop.width + 60, last_input_ontop.height - 30),
                          str("Steer " + "%.3f" % steer), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 270, last_input_ontop.height - 30),
                          str("Throttle " + "%.3f" % throttle), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 480, last_input_ontop.height - 30),
                          str("Brake " + "%.3f" % brake), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 675, last_input_ontop.height - 30),
                              str("Speed " + "%.3f" % inputs_data[-1]['SPEED'][1]['speed']), fill=(255, 255, 255), font=font)
            mat = mat.resize((650, 225))
            mat.save(os.path.join(self.attention_save_path, str(self.att_count).zfill(6) + '.png'))

            data = inputs_data[-1]['can_bus'][1]
            speed_data = inputs_data[-1]['SPEED'][1]

            ## TODO: HARDCODING
            for key, value in speed_data.items():
                if key in data.keys():  # If it exist, add it
                    data[key].update(value)
                else:
                    data.update({key: value})

            with open(os.path.join(self.attention_save_path, 'can_bus' + str(self.att_count).zfill(6) + '.json'), 'w') as fo:
                jsonObj = {}
                jsonObj.update(data)
                fo.seek(0)
                fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
                fo.close()

            self.att_count+=1

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
        self.attention_save_path = None

        self.wallclock_t0 = None
        self.writer = None
        self.att_count = 0

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()
        input_data = self.adding_roach_data(input_data)

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        #if int(wallclock_diff / 60) % 10 == 0:
        #print('======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp/(wallclock_diff+0.001)))

        if len(self.inputs_buffer.queue) <= 0:
            print('=== The agent is stopping and waitting for the input buffer ...')
            self.inputs_buffer.put(input_data)
            return self.stopping_and_wait()

        else:
            inputs = [list(self.inputs_buffer.queue)[i] for i in range(0, len(self.inputs_buffer.queue))]

            control = self.run_step(inputs)
            control.manual_gear_shift = False

            # We pop the first frame of the input buffer and stack the current frame to the end
            self.inputs_buffer.get()
            self.inputs_buffer.put(input_data)

            return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._route_plan = global_plan_world_coord


    def process_command(self, gps, imu):
        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
        _, _, cmd = self.waypointer.tick(gps, imu)

        return encode_directions_4(cmd.value), cmd.value

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
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from _utils.training_utils import DataParallelWrapper
from dataloaders.transforms import encode_directions, inverse_normalize
from leaderboard.utils.waypointer import Waypointer
from leaderboard.envs.data_writer import Writer


def checkpoint_parse_configuration_file(filename):

    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

def get_entry_point():
    return 'FramesStacking_SpeedInput_agent'

class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class FramesStacking_SpeedInput_agent(object):

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
        self.inputs_buffer = Queue()
        self.waypointer = None
        self.attention_save_path = None

        # agent's initialization
        self.setup(path_to_conf_file)

        self.wallclock_t0 = None

        if save_sensor is not None:
            self.writer = Writer(save_sensor)
        else:
            self.writer = None

        self.wallclock_t0 = None

        if save_attention:
            self.attention_save_path = save_attention
            self.att_count = 0

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """

        exp_dir = os.path.join('/', os.path.join(*path_to_conf_file.split('/')[:-1]))
        yaml_conf, checkpoint_number, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        g_conf.immutable(False)
        merge_with_yaml(os.path.join(exp_dir, yaml_conf), process_type='drive')
        set_type_of_process('drive', root=os.environ["TRAINING_RESULTS_ROOT"])

        if g_conf.MODEL_TYPE in ['FramesStacking_SpeedLossInput']:
            self._model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
            if torch.cuda.device_count() > 1 and g_conf.DATA_PARALLEL:
                print("Using multiple GPUs parallel! ")
                print(torch.cuda.device_count(), 'GPUs to be used: ', os.environ["CUDA_VISIBLE_DEVICES"])
                self._model = DataParallelWrapper(self._model)
            self.checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', self._model.name+'_'+str(checkpoint_number) + '.pth'))
            print(self._model.name+'_'+str(checkpoint_number) + '.pth', "loaded from ", os.path.join(exp_dir, 'checkpoints'))
            self._model.load_state_dict(self.checkpoint['model'])
            self._model.cuda()
            self._model.eval()
        else:
            raise RuntimeError('MODEL_TYPE not defined yet!')

    def set_world(self, world):
        self.world=world
        self.map=self.world.get_map()

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        ## This is the default setting
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
             'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_central'},

            #{'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': -15.0, 'yaw': -30.0,
            # 'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_left'},

            #{'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': -15.0, 'yaw': 30.0,
            # 'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_right'},

            {'type': 'sensor.other.gnss', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'id': 'GPS'},

            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},

            {'type': 'sensor.can_bus', 'reading_frequency': 20, 'id': 'can_bus'}
        ]

        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 600, 'height': 170, 'fov': 100, 'id': 'rgb_central'},

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

        control = carla.VehicleControl()
        norm_rgb = [self.process_image(inputs_data[i]['rgb_central'][1]).unsqueeze(0).cuda() for i in range(len(inputs_data))]
        norm_speed = [torch.cuda.FloatTensor([self.process_speed(inputs_data[i]['SPEED'][1]['speed'])]).unsqueeze(0) for i in range(len(inputs_data))]
        direction = [torch.cuda.FloatTensor(self.process_command(inputs_data[i]['GPS'][1], inputs_data[i]['IMU'][1])[0]).unsqueeze(0) for i in range(len(inputs_data))]
        
        actions_outputs, attention_layers,_ = self._model.forward_eval(norm_rgb, direction, norm_speed)
        steer, throttle, brake = self.process_control_outputs(actions_outputs[:, -len(g_conf.TARGETS):].detach().cpu().numpy().squeeze(0))
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False

        if self.attention_save_path:
            if not os.path.exists(self.attention_save_path+'_backbone_attn'):
                os.makedirs(self.attention_save_path+'_backbone_attn')
            if not os.path.exists(self.attention_save_path):
                os.makedirs(self.attention_save_path)

            last_input = inverse_normalize(norm_rgb[-1], g_conf.IMG_NORMALIZATION['mean'], g_conf.IMG_NORMALIZATION['std']).squeeze().cpu().data.numpy()
            cmap_2 = plt.get_cmap('jet')

            att = np.delete(cmap_2(np.abs(attention_layers[-1][-1][0].cpu().data.numpy()).mean(0) / np.abs(
                attention_layers[-1][-1][0].cpu().data.numpy()).mean(0).max()), 3, 2)
            att = np.array(Image.fromarray((att * 255).astype(np.uint8)).resize(
                    (g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])))
            last_att = Image.fromarray(att)
            input = last_input.transpose(1, 2, 0) * 255
            input = input.astype(np.uint8)
            input = Image.fromarray(input)
            blend_im = Image.blend(input, last_att, 0.7)

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

            command_sign = command_sign.resize((130, 40))

            #blend_im.paste(command_sign, (180, 10), command_sign)
            #blend_im.paste(command_sign, (230, 10), command_sign)
            input.paste(command_sign, (230, 10), command_sign)

            #draw = ImageDraw.Draw(blend_im)
            draw = ImageDraw.Draw(input)

            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 20)

            draw.text((46, 150), str("Steer " + "%.3f" % steer), fill=(0, 0, 255), font=font)
            draw.text((185, 150), str("Throttle " + "%.3f" % throttle), fill=(0, 0, 255), font=font)
            draw.text((320, 150), str("Brake " + "%.3f" % brake), fill=(0, 0, 255), font=font)
            draw.text((450, 150), str("Speed " + "%.3f" % inputs_data[-1]['SPEED'][1]['speed']), fill=(0, 0, 255), font=font)

            blend_im.save(os.path.join(self.attention_save_path+'_backbone_attn', str(self.att_count).zfill(6) + '.png'))
            input.save(os.path.join(self.attention_save_path, str(self.att_count).zfill(6) + '.png'))
            self.att_count += 1

        return control

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
        self.checkpoint = None
        self.world = None
        self.map = None

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

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        print('======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp/(wallclock_diff+0.001)))

        if len(self.inputs_buffer.queue) <= ((g_conf.ENCODER_INPUT_FRAMES_NUM - 1) * g_conf.ENCODER_STEP_INTERVAL):
            print('=== The agent is stopping and waitting for the input buffer ...')
            self.inputs_buffer.put(input_data)
            return self.stopping_and_wait()

        else:
            inputs = [list(self.inputs_buffer.queue)[i] for i in range(0, len(self.inputs_buffer.queue), g_conf.ENCODER_STEP_INTERVAL)]
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

    def process_image(self, image):
        image = Image.fromarray(image)
        image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1]))
        image = TF.to_tensor(image)
        # Normalization is really necessary if you want to use any pretrained weights.
        image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
        return image

    def process_speed(self, speed):
        norm_speed = abs(speed - g_conf.DATA_NORMALIZATION['speed'][0]) / (
                g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0])  # [0.0, 1.0]
        return norm_speed

    def process_control_outputs(self, action_outputs):
        if g_conf.ACCELERATION_AS_ACTION:
            steer, acceleration = action_outputs[0], action_outputs[1]
            if acceleration >= 0.0:
                throttle = acceleration
                brake = 0.0
            else:
                brake = np.abs(acceleration)
                throttle = 0.0
        else:
            steer, throttle, brake = action_outputs[0], action_outputs[1], action_outputs[2]
            if brake < 0.05:
                brake = 0.0

        return np.clip(steer, -1, 1), np.clip(throttle, 0, 1), np.clip(brake, 0, 1)

    def process_command(self, gps, imu):
        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
        _, _, cmd = self.waypointer.tick(gps, imu)

        return encode_directions(cmd.value), cmd.value


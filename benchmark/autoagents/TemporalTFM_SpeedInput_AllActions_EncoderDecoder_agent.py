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
from queue import Queue
import json
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from benchmark.utils.route_manipulation import downsample_route
from benchmark.envs.sensor_interface import SensorInterface

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from _utils.training_utils import DataParallelWrapper
from dataloaders.transforms import encode_directions_4, encode_directions_6,inverse_normalize, decode_directions_4, decode_directions_6
from benchmark.utils.waypointer import Waypointer

from pytorch_grad_cam.pytorch_grad_cam.grad_cam import GradCAM

def checkpoint_parse_configuration_file(filename):

    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

def get_entry_point():
    return 'TemporalTFM_SpeedInput_AllActions_EncoderDecoder_agent'

class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class TemporalTFM_SpeedInput_AllActions_EncoderDecoder_agent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, save_driving_vision, save_driving_measurement):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.inputs_buffer=Queue()
        self.waypointer = None
        self.first_run = True
        self.vision_save_path = save_driving_vision
        self.measurement_save_path = save_driving_measurement

        # agent's initialization
        self.setup_model(path_to_conf_file)

        self.cmap_2 = plt.get_cmap('jet')
        self.datapoint_count = 0

    def setup_model(self, path_to_conf_file):
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
        self._model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        if torch.cuda.device_count() > 1 and g_conf.DATA_PARALLEL:
            print("Using multiple GPUs parallel! ")
            print(torch.cuda.device_count(), 'GPUs to be used: ', os.environ["CUDA_VISIBLE_DEVICES"])
            self._model = DataParallelWrapper(self._model)
        self.checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', self._model.name+'_'+str(checkpoint_number) + '.pth'))
        print(self._model.name+'_'+str(checkpoint_number) + '.pth', "loaded from ", os.path.join(exp_dir, 'checkpoints'))
        self._model.load_state_dict(self.checkpoint['model'])
        self._model.cuda()
        self._model.share_memory()
        self._model.eval()

        if g_conf.ACCELERATION_AS_ACTION:
            self.latest_actions = [[0.0, 1.0]]
        else:
            self.latest_actions = [[0.0, 1.0, 0.0]]

    def set_world(self, world):
        self.world=world
        self.map=self.world.get_map()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 10000000)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in
                                         ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self.waypointer = Waypointer(self._global_plan, self._global_plan[0][0], self.world)
    
    def set_ego_vehicle(self, ego_vehicle):
        self._ego_vehicle=ego_vehicle

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

        for key, values in input_data.items():
            if values[0] != timestamp.frame:
                raise RuntimeError(' The frame number of sensor data does not match the timestamp frame:', key)

        if len(self.inputs_buffer.queue) < ((g_conf.ENCODER_INPUT_FRAMES_NUM-1) * g_conf.ENCODER_STEP_INTERVAL):
            print('=== The agent is stopping and waitting for the input buffer ...')

            if g_conf.ACCELERATION_AS_ACTION:
                input_data.update({'actions': [[0.0, 1.0]]})
            else:
                input_data.update({'actions': [[0.0, 1.0, 0.0]]})

            self.inputs_buffer.put(input_data)
            return self.stopping_and_wait()

        else:
            # we add a fake current's action (no use in the model), and stack the current inputs to the end of buffer
            input_data.update({'actions': self.latest_actions})
            self.inputs_buffer.put(input_data)

            inputs = [list(self.inputs_buffer.queue)[i] for i in range(0, len(self.inputs_buffer.queue), g_conf.ENCODER_STEP_INTERVAL)]
            control, self.latest_actions = self.run_step(inputs)
            control.manual_gear_shift = False

            # we update the current's actions by the prediction
            input_data.update({'actions': self.latest_actions})
            self.inputs_buffer.queue[-1] = input_data

            # We pop the first frame of the input buffer
            self.inputs_buffer.get(True)

            return control

    def run_step(self, inputs_data):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        norm_rgb = [self.process_image(inputs_data[i]['rgb_central'][1]).unsqueeze(0).cuda() for i in range(len(inputs_data))]
        if self.first_run:
            self.direction = [torch.cuda.FloatTensor(self.process_command(inputs_data[i]['GPS'][1], inputs_data[i]['IMU'][1])[0]).unsqueeze(0).cuda()
                         for i in range(len(inputs_data))]
            self.first_run=False
        else:
            self.direction = self.direction[-4:] + \
                             [torch.cuda.FloatTensor(self.process_command(inputs_data[-1]['GPS'][1], inputs_data[-1]['IMU'][1])[0]).unsqueeze(0).cuda()]
        norm_speed = [torch.cuda.FloatTensor([self.process_speed(inputs_data[i]['SPEED'][1]['speed'])]).unsqueeze(0).cuda() for i in range(len(inputs_data))]
        actions = [torch.cuda.FloatTensor(inputs_data[i]['actions']).cuda() for i in range(len(inputs_data))]
        actions_outputs, att_backbone_layers, attn_weights, _, _ = self._model.forward_eval(norm_rgb, self.direction, norm_speed, actions)
        all_action_outputs = [
            self.process_control_outputs(actions_outputs[:, i, -len(g_conf.TARGETS):].detach().cpu().numpy().squeeze(0))
            for i in range(g_conf.ENCODER_INPUT_FRAMES_NUM)]

        steer, throttle, brake = all_action_outputs[-1]
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False

        if self.vision_save_path:
            if not os.path.exists(self.vision_save_path):
                os.makedirs(self.vision_save_path)
            rgb_img = inverse_normalize(norm_rgb[-1], g_conf.IMG_NORMALIZATION['mean'],
                                        g_conf.IMG_NORMALIZATION['std']).squeeze().cpu().data.numpy()
            last_input = rgb_img.transpose(1, 2, 0) * 255
            last_input = last_input.astype(np.uint8)
            last_input = Image.fromarray(last_input)

            target_layers = [self._model._model.encoder_embedding_perception.layer4[-1]]
            cam = GradCAM(model=self._model._model.encoder_embedding_perception, target_layers=target_layers)
            input_tensor = norm_rgb[-1]
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            att = np.delete(self.cmap_2(grayscale_cam), 3, 2)
            att = (att * 255).astype(np.uint8)
            last_att = Image.fromarray(att)

            blend_im = Image.blend(last_input, last_att, 0.5)

            last_input_ontop = Image.fromarray(inputs_data[-1]['rgb_ontop'][1])

            cmd = decode_directions_4(self.direction[-1].detach().cpu().numpy().squeeze(0))
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

            command_sign = command_sign.resize((130, 40))

            mat = Image.new('RGB', (
                last_input_ontop.width + last_input.width, max(last_input_ontop.height, last_input.height * 2)),
                            (0, 0, 0))
            mat.paste(command_sign, (last_input_ontop.width + 230, last_input.height * 2 + 20))

            mat.paste(last_input_ontop, (0, 0))
            mat.paste(last_input, (last_input_ontop.width, 0))
            mat.paste(blend_im, (last_input_ontop.width, last_input.height))

            draw_mat = ImageDraw.Draw(mat)
            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 20)
            draw_mat.text((last_input_ontop.width + 40, last_input_ontop.height - 30),
                          str("Steer " + "%.3f" % steer), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 180, last_input_ontop.height - 30),
                          str("Throttle " + "%.3f" % throttle), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 320, last_input_ontop.height - 30),
                          str("Brake " + "%.3f" % brake), fill=(255, 255, 255), font=font)
            draw_mat.text((last_input_ontop.width + 450, last_input_ontop.height - 30),
                              str("Speed " + "%.3f" % inputs_data[-1]['SPEED'][1]['speed']), fill=(255, 255, 255), font=font)

            #mat = mat.resize((420, 180))
            mat.save(os.path.join(self.vision_save_path, str(self.datapoint_count).zfill(6) + '.jpg'))

        if self.measurement_save_path:
            if not os.path.exists(self.measurement_save_path):
                os.makedirs(self.measurement_save_path)

            # we record the driving measurement data
            data = inputs_data[-1]['can_bus'][1]
            data.update({'steer': np.nan_to_num(control.steer)})
            data.update({'throttle': np.nan_to_num(control.throttle)})
            data.update({'brake': np.nan_to_num(control.brake)})
            data.update({'hand_brake': control.hand_brake})
            data.update({'reverse': control.reverse})
            data.update({'speed': inputs_data[-1]['SPEED'][1]})

            with open(os.path.join(self.measurement_save_path, 'can_bus' + str(self.datapoint_count).zfill(6) + '.json'), 'w') as fo:
                jsonObj = {}
                jsonObj.update(data)
                fo.seek(0)
                fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
                fo.close()

        self.datapoint_count += 1

        last_action_outputs = actions_outputs[:, -1, -len(g_conf.TARGETS):].detach().cpu().numpy()

        return control, last_action_outputs

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
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = None
        self.inputs_buffer = None
        self.waypointer = None
        self.vision_save_path = None
        self.measurement_save_path = None
        self.datapoint_count = 0

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
        _, _, cmd = self.waypointer.tick(gps, imu)

        if g_conf.DATA_COMMAND_CLASS_NUM == 4:
            return encode_directions_4(cmd.value), cmd.value
        elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
            return encode_directions_6(cmd.value), cmd.value



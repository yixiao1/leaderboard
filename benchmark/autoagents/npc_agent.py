#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
import numpy as np
import os
import json
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.envs.sensor_interface import SensorInterface
from leaderboard.envs.data_writer import Writer


def get_entry_point():
    return 'NpcAgent'

class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file, save_sensor=None):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._route_assigned = False
        self._agent = None

        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None

        if save_sensor is not None:
            self.writer = Writer(save_sensor)
        else:
            self.writer = None

    def set_world(self, world):
        self.world=world
        self.map=self.world.get_map()

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """

        #sensors = [
        #    {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
        #     'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_central'},

        #    {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': -15.0, 'yaw': -30.0,
        #     'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_left'},

        #    {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'roll': 0.0, 'pitch': -15.0, 'yaw': 30.0,
        #     'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_right'},

        #    {'type': 'sensor.other.gnss', 'x': 2.0, 'y': 0.0, 'z': 1.40, 'id': 'GPS'},

        #    {'type': 'sensor.other.imu', 'id': 'IMU'},

        #    {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},

        #    {'type': 'sensor.can_bus', 'reading_frequency': 20, 'id': 'can_bus'}
        #]

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 900, 'height': 256, 'fov': 100, 'id': 'rgb_central'},

            {'type': 'sensor.other.gnss', 'id': 'GPS'},

            {'type': 'sensor.other.imu', 'id': 'IMU'},

            {'type': 'sensor.speedometer', 'id': 'SPEED'},

            {'type': 'sensor.can_bus', 'id': 'can_bus'}
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor)

            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                prev = None
                for transform, _ in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    if  prev:
                        route_segment = self._agent._trace_route(prev, wp)
                        plan.extend(route_segment)

                    prev = wp

                #loc = plan[-1][0].transform.location
                #self._agent.set_destination([loc.x, loc.y, loc.z])
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True

        else:
            control = self._agent.run_step()

        return control

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

        control = self.run_step(input_data, timestamp)

        control.manual_gear_shift = False

        return control

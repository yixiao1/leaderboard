#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import xml.etree.ElementTree as ET
import numpy.random as random

import py_trees
import os
import matplotlib.pyplot as plt

import carla
import time

from agents.navigation.local_planner import RoadOption

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.scenarios.control_loss_YiXiao import ControlLoss
from srunner.scenarios.follow_leading_vehicle_YiXiao import FollowLeadingVehicleWithObstacle
from srunner.scenarios.object_crash_vehicle_YiXiao import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection_YiXiao import VehicleTurningRoute
from srunner.scenarios.selfDefined_scenarios_YiXiao import (SignalJunctionLeadingVehicleCrossingTrafficLight,
                                                            SignalJunctionGreenTrafficLightObstacleCrossing)


from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)

from benchmark.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from benchmark.utils.route_manipulation import interpolate_trajectory
from tools.utils import draw_map, draw_point_data

ROUTESCENARIO = ["RouteScenario"]

SECONDS_GIVEN_PER_METERS = 0.8
INITIAL_SECONDS_DELAY = 5.0

SELFDEFINED_NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicleWithObstacle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": SignalJunctionLeadingVehicleCrossingTrafficLight,
    "Scenario6": SignalJunctionGreenTrafficLightObstacleCrossing
}


def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot.
    """
    # Initialize the variables
    blackboard = py_trees.blackboard.Blackboard()
    _ = blackboard.set(variable_name, False)

    # Wait until the scenario has ended
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(
        name=variable_name + " Done?",
        variable_name=variable_name,
        expected_value=True,
        clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
    )
    set_flag = py_trees.blackboard.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot")
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """
    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"

    def __init__(self, world, config, debug_mode=0, criteria_enable=True):
        """
        Setup all relevant parameters and create scenarios along route
        """
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None

        self._update_route(world, config, debug_mode>0)

        ego_vehicle = self._update_ego_vehicle()

        self.list_scenarios = self._build_scenario_instances(world,
                                                             ego_vehicle,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=10,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode>1)

        super(RouteScenario, self).__init__(name=config.name,
                                            ego_vehicles=[ego_vehicle],
                                            config=config,
                                            world=world,
                                            debug_mode=debug_mode>1,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """

        # Transform the scenario file into a dictionary
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

        # prepare route's trajectory (interpolate and add the GPS route)
        gps_route, route = interpolate_trajectory(world, config.trajectory)

        print(' ')
        print('route name', config.name)
        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(
            config, route, world_annotations)

        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        print(' ')
        print('sampled_scenarios_definitions')
        print(self.sampled_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout()

        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

        if True:
            save_path = os.path.join(os.environ['SENSOR_SAVE_PATH'], config.package_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            trajectories_fig = plt.figure(0)
            if config.town == 'Town01':
                plt.xlim(-500, 5000)
                plt.ylim(-500, 4500)
            elif config.town == 'Town02':
                plt.xlim(-250, 2500)
                plt.ylim(1000, 4000)

            draw_map(world)

            count=0
            for wp in self.route:
                datapoint = [wp[0].location.x, wp[0].location.y, wp[0].location.z]
                draw_point_data(datapoint, trajectories_fig)

            for wp_tj in config.trajectory:
                datapoint = [wp_tj.x, wp_tj.y, wp_tj.z]
                if count == 0:
                        draw_point_data(datapoint, trajectories_fig, color=(0/ 255.0, 0/ 255.0, 255/ 255.0), size=30, text=(round(wp_tj.x,2), round(wp_tj.y, 2)))
                else:
                    if count == len(config.trajectory)-1:
                        draw_point_data(datapoint, trajectories_fig, color=(252/ 255.0, 175/ 255.0, 62/ 255.0), size=20, text=(round(wp_tj.x,2), round(wp_tj.y, 2)))
                    else:
                        draw_point_data(datapoint, trajectories_fig, color=(252 / 255.0, 175 / 255.0, 62 / 255.0), size=20)
                count += 1

            for scenarios_definition in self.sampled_scenarios_definitions:
                trigger_position=scenarios_definition['trigger_position']
                datapoint = [trigger_position['x'], trigger_position['y'], trigger_position['z']]
                draw_point_data(datapoint, trajectories_fig, color=(0.0/ 255.0, 255.0/ 255.0, 0.0/ 255.0), size=20, text=(trigger_position['x'], trigger_position['y'], trigger_position['yaw']))

            trajectories_fig.savefig(os.path.join(save_path, config.name + '.png'),
                                     orientation='landscape', bbox_inches='tight', dpi=1200)
            plt.close(trajectories_fig)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz_2017',
                                                          elevate_transform,
                                                          rolename='hero')

        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        # we change the Vehicle Physics Control the same as the setting in CARLA 0.9.10
        vehicle_physics = ego_vehicle.get_physics_control()

        """

        print( '================================= Before ================================== ')
        print("\n torque_curve", [[val.x, val.y] for val in vehicle_physics.torque_curve], \
              "\n max_rpm", vehicle_physics.max_rpm, \
              "\n moi", vehicle_physics.moi, \
              "\n damping_rate_full_throttle", vehicle_physics.damping_rate_full_throttle, \
              "\n damping_rate_zero_throttle_clutch_engaged", vehicle_physics.damping_rate_zero_throttle_clutch_engaged, \
              "\n damping_rate_zero_throttle_clutch_disengaged",
              vehicle_physics.damping_rate_zero_throttle_clutch_disengaged, \
              "\n use_gear_autobox", vehicle_physics.use_gear_autobox, \
              "\n gear_switch_time", vehicle_physics.gear_switch_time, \
              "\n clutch_strength", vehicle_physics.clutch_strength, \
              "\n final_ratio", vehicle_physics.final_ratio)

        for i, gear in enumerate(vehicle_physics.forward_gears):
            print("gear number", i, '\n', "gear.ratio", gear.ratio, 'gear.down_ratio', gear.down_ratio, "gear.up_ratio",
                  gear.up_ratio)

        print("\n mass", vehicle_physics.mass, \
              "\n drag_coefficient", vehicle_physics.drag_coefficient, \
              "\n center_of_mass", vehicle_physics.center_of_mass)
        print("\n steering_curve", [[val.x, val.y] for val in vehicle_physics.steering_curve], \
              # "\n use_sweep_wheel_collision", vehicle_physics.use_sweep_wheel_collision)
              "\n use_sweep_wheel_collision", 'NoImplement')

        for wheel, ids in zip(vehicle_physics.wheels, ['front left', 'front right', 'back left', 'back right']):
            print('\n \n', ids, \
                  '\n tire_friction', wheel.tire_friction, \
                  '\n damping_rate', wheel.damping_rate, \
                  '\n max_steer_angle', wheel.max_steer_angle, \
                  '\n radius', wheel.radius, \
                  '\n max_brake_torque', wheel.max_brake_torque, \
                  '\n max_handbrake_torque', wheel.max_handbrake_torque, \
                  '\n position', wheel.position, \
                  # '\n long_stiff_value', wheel.long_stiff_value, \
                  '\n long_stiff_value', 'NoImplement', \
                  # '\n lat_stiff_max_load', wheel.lat_stiff_max_load, \
                  '\n lat_stiff_max_load', 'NoImplement', \
                  # '\n lat_stiff_value', wheel.lat_stiff_value)
                  '\n lat_stiff_value', 'NoImplement')

        print("\n ===================================================")
        
        """

        vehicle_physics.torque_curve = [carla.Vector2D(x=0, y=400),
                                        carla.Vector2D(x=1890, y=500),
                                        carla.Vector2D(x=5730, y=400)]
        vehicle_physics.max_rpm = 5800
        vehicle_physics.damping_rate_full_throttle = 0.15
        vehicle_physics.gear_switch_time = 0.5
        vehicle_physics.final_ratio = 4
        vehicle_physics.mass = 2404
        front_left_wheel = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=70.0, radius=35.5,
                                                     max_brake_torque=1500.0, max_handbrake_torque=0.0,
                                                     position=carla.Vector3D(x=-1855.752686,y=20321.986328,z=35.430557))
        front_right_wheel = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=70.0, radius=35.5,
                                                     max_brake_torque=1500.0, max_handbrake_torque=0.0,
                                                     position=carla.Vector3D(x=-1855.752686,y=20478.412109,z=35.430561))
        rear_left_wheel = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=0.0, radius=35.5,
                                                     max_brake_torque=1500.0, max_handbrake_torque=3000.0,
                                                     position=carla.Vector3D(x=-2142.864746, y=20322.703125, z=35.430557))
        rear_right_wheel = carla.WheelPhysicsControl(tire_friction=3.5, damping_rate=0.25, max_steer_angle=0.0, radius=35.5,
                                                     max_brake_torque=1500.0, max_handbrake_torque=3000.0,
                                                     position=carla.Vector3D(x=-2142.474365, y=20479.128906, z=35.430561))

        vehicle_physics.wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]

        time.sleep(3)
        ego_vehicle.apply_physics_control(vehicle_physics)
        time.sleep(3)

        vehicle_physics_new = ego_vehicle.get_physics_control()

        print('===================== After modification =====================')
        print("\n mass", vehicle_physics_new.mass, 'shoud be 2404')
        print("\n damping_rate_full_throttle", vehicle_physics.damping_rate_full_throttle, 'shoud be 0.15')
        print('\n ===============================================')

        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0) # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        def select_scenario(list_scenarios):
            # priority to the scenarios with higher number: 10 has priority over 9, etc.
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1

                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario

            return selected_scenario

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = select_scenario(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []

        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'],
                                     scenario['trigger_position']['y'],
                                     scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                        color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number
            scenario_class = SELFDEFINED_NUMBER_CLASS_TRANSLATION[definition['name']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position

            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                          ego_vehicle.get_transform(),
                                                                          'hero')]
            scenario_configuration.route = self.route
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name
            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                   criteria_enable=False, timeout=timeout)
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

            except Exception as e:
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Create the background activity of the route
        town_amount = {
            'Town01': 120,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120,
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10': 120,
        }

        scenario_amount = {
            'Scenario1': 0,
            'Scenario2': 0,
            'Scenario3': 0,
            'Scenario4': 0,
            'Scenario5': 0,
            'Scenario6': 0,
        }

        #amount = town_amount[config.town] if config.town in town_amount else 0
        amount = scenario_amount[config.defined_available_senarios_list[0]] if config.defined_available_senarios_list[0] in scenario_amount else 0

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                   policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        scenario_behaviors = []
        blackboard_list = []

        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name

                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name,
                                            scenario.config.trigger_points[0].location])
                else:
                    name = "{} - {}".format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(
                        name=name,
                        variable_name=name,
                        behaviour=scenario.scenario.behavior)
                    scenario_behaviors.append(oneshot_idiom)

        # Add behavior that manages the scenarios trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0],
            self.route,
            blackboard_list,
            scenario_trigger_distance,
            repeat_scenarios=False
        )

        subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria = []
        route = convert_transform_to_location(self.route)

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=True)

        route_criterion = InRouteTest(self.ego_vehicles[0],
                                      route=route,
                                      offroad_max=30,
                                      terminate_on_failure=True)
                                      
        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route, terminate_on_failure=False)

        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route, terminate_on_failure=True)

        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0], terminate_on_failure=False)

        stop_criterion = RunningStopTest(self.ego_vehicles[0], terminate_on_failure=False)

        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0],
                                                         speed_threshold=0.1,
                                                         below_threshold_max_time=180.0,
                                                         terminate_on_failure=True,
                                                         name="AgentBlockedTest")

        criteria.append(completion_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(collision_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(route_criterion)
        criteria.append(blocked_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

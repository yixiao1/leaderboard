#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import math

import carla
import numpy as np
import math
import py_trees
from collections import OrderedDict

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenarios.basic_scenario import BasicScenario

from agents.navigation.local_planner import RoadOption

DEBUG_COLORS = {
    'road': carla.Color(0,0, 255),      # Blue
    'opposite': carla.Color(255,0, 0),  # Red
    'junction': carla.Color(0, 0, 0),   # Black
    'entry': carla.Color(255,255, 0),   # Yellow
    'exit': carla.Color(0,255, 255),    # Teal
    'connect': carla.Color(0,255, 0),   # Green
}

DEBUG_TYPE = {
    'small': [0.8, 0.1],
    'medium': [0.5, 0.15],
    'large': [0.2, 0.2],
}

def draw_string(world, location, string='', type='road', persistent=False):
    """Utility function to draw debugging strings"""
    v_shift, _ = DEBUG_TYPE.get('small')
    l_shift = carla.Location(z=v_shift)
    color = DEBUG_COLORS.get(type, 'road')
    life_time = 0.07 if not persistent else 100000
    world.debug.draw_string(location + l_shift, string, False, color, life_time)

def draw_point(world, location, point_type='small', type='road', persistent=False):
    """Utility function to draw debugging points"""
    v_shift, size = DEBUG_TYPE.get(point_type, 'small')
    l_shift = carla.Location(z=v_shift)
    color = DEBUG_COLORS.get(type, 'road')
    life_time = 0.07 if not persistent else 100000
    world.debug.draw_point(location + l_shift, size, color, life_time)

def get_same_dir_lanes( waypoint):
    """Gets all the lanes with the same direction of the road of a wp"""
    same_dir_wps = [waypoint]

    # Check roads on the right
    right_wp = waypoint
    while True:
        possible_right_wp = right_wp.get_right_lane()
        if possible_right_wp is None or possible_right_wp.lane_type != carla.LaneType.Driving:
            break
        right_wp = possible_right_wp
        same_dir_wps.append(right_wp)

    # Check roads on the left
    left_wp = waypoint
    while True:
        possible_left_wp = left_wp.get_left_lane()
        if possible_left_wp is None or possible_left_wp.lane_type != carla.LaneType.Driving:
            break
        if possible_left_wp.lane_id * left_wp.lane_id < 0:
            break
        left_wp = possible_left_wp
        same_dir_wps.append(left_wp)

    return same_dir_wps

def get_opposite_dir_lanes(waypoint):
    """Gets all the lanes with opposite direction of the road of a wp"""
    other_dir_wps = []
    other_dir_wp = None

    # Get the first lane of the opposite direction
    left_wp = waypoint
    while True:
        possible_left_wp = left_wp.get_left_lane()
        if possible_left_wp is None:
            break
        if possible_left_wp.lane_id * left_wp.lane_id < 0:
            other_dir_wp = possible_left_wp
            break
        left_wp = possible_left_wp

    if not other_dir_wp:
        return other_dir_wps

    # Check roads on the right
    right_wp = other_dir_wp
    while True:
        if right_wp.lane_type == carla.LaneType.Driving:
            other_dir_wps.append(right_wp)
        possible_right_wp = right_wp.get_right_lane()
        if possible_right_wp is None:
            break
        right_wp = possible_right_wp

    return other_dir_wps

def get_lane_key(wp):
    """Returns a key corresponding to the waypoint lane. Equivalent to a 'Lane'
    object and used to compare waypoint lanes"""
    return '' if wp is None else get_road_key(wp) + '*' + str(wp.lane_id)

def get_road_key(wp):
    """Returns a key corresponding to the waypoint road. Equivalent to a 'Road'
    object and used to compare waypoint roads"""
    return '' if wp is None else str(wp.road_id)


class Source(object):

    """
    Source object to store its position and its responsible actors
    """

    def __init__(self, wp, actors, entry_lane_wp=''):
        self.wp = wp
        self.actors = actors
        self.entry_lane_wp = entry_lane_wp
        self.previous_lane_keys = []  # Source lane and connecting lanes of the previous junction


class Junction(object):

    """
    Junction object. Stores its topology as well as its state, when active
    """

    def __init__(self, junction, id, route_entry_index=None, route_exit_index=None):
        # Topology
        self.junctions = [junction]
        self.id = id
        self.route_entry_index = route_entry_index
        self.route_exit_index = route_exit_index
        self.exit_road_length = 0
        self.route_entry_keys = []
        self.route_exit_keys = []
        self.opposite_entry_keys = []
        self.opposite_exit_keys = []
        self.entry_wps = []
        self.exit_wps = []
        self.entry_directions = {'ref': [], 'opposite': [], 'left': [], 'right': []}
        self.exit_directions = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # State
        self.entry_sources = []
        self.exit_sources = []
        self.exit_dict = OrderedDict()
        self.actor_dict = OrderedDict()

    def contains(self, other_junction):
        """"""
        other_id = other_junction.id
        for junction in self.junctions:
            if other_id == junction.id:
                return True
        return False


class BackgroundActivity(BasicScenario):

    """
    Implementation of a scenario to spawn a set of background actors,
    and to remove traffic jams in background traffic

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicle, config, route, night_mode=False, debug_mode=False, timeout=0):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.route = route
        self.config = config
        self._night_mode = night_mode
        self.debug = debug_mode
        self.timeout = timeout  # Timeout of scenario in seconds

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                                 [ego_vehicle],
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=True,
                                                 criteria_enable=True)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        # Check if a vehicle is further than X, destroy it if necessary and respawn it
        return BackgroundBehavior(self.ego_vehicle, self.route, self._night_mode)

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        pass

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        pass


class BackgroundBehavior(AtomicBehavior):
    """
    Handles the background activity
    """
    def __init__(self, ego_actor, route, night_mode=False, debug=False, name="BackgroundBehavior"):
        """
        Setup class members
        """
        super(BackgroundBehavior, self).__init__(name)
        self.debug = debug
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._tm.global_percentage_speed_difference(0.0)
        self._night_mode = night_mode

        # Global variables
        self._ego_actor = ego_actor
        self._ego_state = 'road'
        self._get_route_data(route)

        self._spawn_vertical_shift = 0.2
        self._reuse_dist = 10  # When spawning actors, might reuse actors closer to this distance
        self._spawn_free_radius = 30  # Sources closer to the ego will not spawn actors
        self._fake_junction_ids = []
        self._fake_lane_pair_keys = []

        # Road variables
        self._road_actors = []
        self._road_sources = []
        self._road_checker_index = 0

        self._road_front_vehicles = 3  # Amount of vehicles in front of the ego
        self._road_back_vehicles = 3  # Amount of vehicles behind the ego
        self._road_vehicle_dist = 8  # Distance road vehicles leave betweeen each other[m]
        self._road_spawn_dist = 13  # Initial distance between spawned road vehicles [m]
        self._radius_increase_ratio = 2.0  # Meters the radius increases per m/s of the ego
        self._get_road_radius()

        # Junction variables
        self._junctions = []
        self._active_junctions = []

        self._junction_sources_dist = 40  # Distance from the entry sources to the junction [m]
        self._junction_vehicle_dist = 8  # Distance junction vehicles leave betweeen each other[m]
        self._junction_spawn_dist = 10  # Initial distance between spawned junction vehicles [m]
        self._junction_sources_max_actors = 5  # Maximum vehicles alive at the same time per source

        # Opposite lane variables
        self._opposite_actors = []
        self._opposite_sources = []
        self._opposite_route_index = 0

        self._opposite_removal_dist = 30  # Distance at which actors are destroyed
        self._opposite_sources_dist = 60  # Distance from the ego to the opposite sources [m]
        self._opposite_vehicle_dist = 8  # Distance opposite vehicles leave betweeen each other[m]
        self._opposite_spawn_dist = 20  # Initial distance between spawned opposite vehicles [m]
        self._opposite_sources_max_actors = 5  # Maximum vehicles alive at the same time per source

        # Scenario 2 variables
        self._is_scenario_2_active = False
        self._scenario_2_actors = []
        self._activate_break_scenario = False
        self._break_duration = 7  # Duration of the scenario

        # Scenario 4 variables
        self._is_scenario_4_active = False
        self._scenario_4_actors = []
        self._ego_exitted_junction = False
        self._crossing_dist = None  # Distance between the crossing object and the junction exit

        # Junction scenario variables
        self._remove_entries = False  # Remove all actors in the direction entry when triggering the scenario
        self._remove_exits = False  # Remove all actors at the direction exit when triggering the scenario
        self._remove_middle = False  # Remove all actors in the junction when triggering the scenario
        self._direction = None

    def _get_route_data(self, route):
        """Extract the information from the route"""
        self._route = []  # Transform the route into a list of waypoints
        self._accum_dist = []  # Save the total traveled distance for each waypoint
        prev_trans = None
        for trans, _ in route:
            self._route.append(self._map.get_waypoint(trans.location))
            if prev_trans:
                dist = trans.location.distance(prev_trans.location)
                self._accum_dist.append(dist + self._accum_dist[-1])
            else:
                self._accum_dist.append(0)
            prev_trans = trans

        self._route_length = len(route)
        self._route_index = 0
        self._route_buffer = 3

    def _get_road_radius(self):
        """Computes the min and max radius of the road behavior"""
        self._base_min_radius = self._road_front_vehicles * self._road_spawn_dist
        self._base_max_radius = (self._road_front_vehicles + 1) * self._road_spawn_dist
        self._min_radius = self._base_min_radius
        self._max_radius = self._base_max_radius

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        self._create_junction_dict()
        ego_wp = self._route[0]
        same_dir_wps = get_same_dir_lanes(ego_wp)
        self._initialise_road_behavior(same_dir_wps, ego_wp)
        self._initialise_opposite_sources()
        self._initialise_road_checker()

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        prev_ego_index = self._route_index
        # Check if the TM destroyed an actor
        if self._route_index > 0:
            self._check_background_actors()

        # Get ego's odometry. For robustness, the closest route point will be used
        location = CarlaDataProvider.get_location(self._ego_actor)
        ego_wp = self._update_ego_route_location(location)
        ego_transform = ego_wp.transform
        if self.debug:
            string = 'EGO_' + self._ego_state[0].upper()
            draw_string(self._world, location, string, self._ego_state, False)

        # Parameters and scenarios
        self._update_parameters()
        self._manage_break_scenario()

        # Update ego state
        if self._ego_state == 'junction':
            self._monitor_ego_junction_exit(ego_wp)
        self._monitor_nearby_junctions()

        # Update_actors
        if self._ego_state == 'junction':
            self._monitor_ego_junction_exit(ego_wp)
            self._update_junction_actors()
            self._update_junction_sources()
        else:
            self._update_road_actors(ego_transform)
            self._move_road_checker(prev_ego_index, self._route_index)
            self._move_opposite_sources(prev_ego_index, self._route_index)
            self._update_opposite_sources()

        # Update non junction sources
        self._update_opposite_actors(ego_transform)
        self._update_road_sources(ego_transform.location)

        self._monitor_scenario_4_end(ego_transform.location)

        return new_status

    def terminate(self, new_status):
        """Destroy all actors"""
        all_actors = self._get_actors()
        for actor in list(all_actors):
            self._destroy_actor(actor)
        super(BackgroundBehavior, self).terminate(new_status)

    def _get_actors(self):
        """Returns a list of all actors part of the background activity"""
        actors = self._road_actors + self._opposite_actors
        for junction in self._active_junctions:
            actors.extend(list(junction.actor_dict))
        return actors

    def _check_background_actors(self):
        """Checks if the Traffic Manager has removed a backgroudn actor"""
        background_actors = self._get_actors()
        alive_ids = [actor.id for actor in self._world.get_actors().filter('vehicle*')]
        for actor in background_actors:
            if actor.id not in alive_ids:
                self._remove_actor_info(actor)

    ################################
    ##       Junction cache       ##
    ################################

    def _create_junction_dict(self):
        """Extracts the junctions the ego vehicle will pass through."""
        data = self._get_junctions_data()
        fake_data, filtered_data = self._filter_fake_junctions(data)
        self._get_fake_lane_pairs(fake_data)
        route_data = self._join_complex_junctions(filtered_data)
        self._add_junctions_topology(route_data)
        self._junctions = route_data

    def _get_junctions_data(self):
        """Gets all the junctions the ego passes through"""
        junction_data = []
        junction_num = 0
        start_index = 0

        # Ignore the junction the ego spawns at
        for i in range(0, self._route_length - 1):
            if not self._is_junction(self._route[i]):
                start_index = i
                break

        for i in range(start_index, self._route_length - 1):
            next_wp = self._route[i+1]
            prev_junction = junction_data[-1] if len(junction_data) > 0 else None

            # Searching for the junction exit
            if prev_junction and prev_junction.route_exit_index is None:
                if not self._is_junction(next_wp) or next_wp.get_junction().id != junction_id:
                    prev_junction.route_exit_index = i+1

            # Searching for a junction
            elif self._is_junction(next_wp):
                junction_id = next_wp.get_junction().id
                if prev_junction:
                    start_dist = self._accum_dist[i]
                    prev_end_dist = self._accum_dist[prev_junction.route_exit_index]
                    prev_junction.exit_road_length = start_dist - prev_end_dist

                # Same junction as the prev one and closer than 2 meters
                if prev_junction and prev_junction.junctions[-1].id == junction_id:
                    start_dist = self._accum_dist[i]
                    prev_end_dist = self._accum_dist[prev_junction.route_exit_index]
                    distance = start_dist - prev_end_dist
                    if distance < 2:
                        prev_junction.junctions.append(next_wp.get_junction())
                        prev_junction.route_exit_index = None
                        continue

                junction_data.append(Junction(next_wp.get_junction(), junction_num, i))
                junction_num += 1

        if len(junction_data) > 0:
            road_end_dist = self._accum_dist[self._route_length - 1]
            if junction_data[-1].route_exit_index:
                route_start_dist = self._accum_dist[junction_data[-1].route_exit_index]
            else:
                route_start_dist = self._accum_dist[self._route_length - 1]
            junction_data[-1].exit_road_length = road_end_dist - route_start_dist

        return junction_data

    def _filter_fake_junctions(self, data):
        """
        Filters fake junctions. As a general note, a fake junction is that where no road lane divide in two.
        However, this might fail for some CARLA maps, so check junctions which have all lanes straight too
        """
        fake_data = []
        filtered_data = []
        threshold = math.radians(15)

        for junction_data in data:
            used_entry_lanes = []
            used_exit_lanes = []
            for junction in junction_data.junctions:
                for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving): 
                    entry_wp = self._get_junction_entry_wp(entry_wp)
                    if not entry_wp:
                        continue
                    if get_lane_key(entry_wp) not in used_entry_lanes:
                        used_entry_lanes.append(get_lane_key(entry_wp))

                    exit_wp = self._get_junction_exit_wp(exit_wp)
                    if not exit_wp:
                        continue
                    if get_lane_key(exit_wp) not in used_exit_lanes:
                        used_exit_lanes.append(get_lane_key(exit_wp))

            if not used_entry_lanes and not used_exit_lanes:
                fake_data.append(junction_data)

            found_turn = False
            for entry_wp, exit_wp in junction_data.junctions[0].get_waypoints(carla.LaneType.Driving):
                entry_heading = entry_wp.transform.get_forward_vector()
                exit_heading = exit_wp.transform.get_forward_vector()
                dot = entry_heading.x * exit_heading.x + entry_heading.y * exit_heading.y
                if dot < math.cos(threshold):
                    found_turn = True
                    break

            if not found_turn:
                fake_data.append(junction_data)
            else:
                filtered_data.append(junction_data)

        return fake_data, filtered_data

    def _get_complex_junctions(self):
        """
        Function to hardcode the topology of some complex junctions. This is done for the roundabouts,
        as the current API doesn't offer that info as well as others such as the gas station at Town04.
        If there are micro lanes between connected junctions, add them to the fake_lane_keys, connecting
        them when their topology is calculated
        """
        complex_junctions = []
        fake_lane_keys = []

        if 'Town03' in self._map.name:
            # Roundabout, take it all as one
            complex_junctions.append([
                self._map.get_waypoint_xodr(1100, -5, 16.6).get_junction(),
                self._map.get_waypoint_xodr(1624, -5, 25.3).get_junction(),
                self._map.get_waypoint_xodr(1655, -5, 8.3).get_junction(),
                self._map.get_waypoint_xodr(1772, 3, 16.2).get_junction(),
                self._map.get_waypoint_xodr(1206, -5, 5.9).get_junction()])
            fake_lane_keys.extend([
                ['37*-4','36*-4'], ['36*-4','37*-4'],
                ['37*-5','36*-5'], ['36*-5','37*-5'],
                ['38*-4','12*-4'], ['12*-4','38*-4'],
                ['38*-5','12*-5'], ['12*-5','38*-5']])

            # Gas station
            complex_junctions.append([
                self._map.get_waypoint_xodr(1031, -1, 11.3).get_junction(),
                self._map.get_waypoint_xodr(100, -1, 18.8).get_junction(),
                self._map.get_waypoint_xodr(1959, -1, 22.7).get_junction()])
            fake_lane_keys.extend([
                ['32*-2','33*-2'], ['33*-2','32*-2'],
                ['32*-1','33*-1'], ['33*-1','32*-1'],
                ['32*4','33*4'], ['33*4','32*4'],
                ['32*5','33*5'], ['33*5','32*5']])

        elif 'Town04' in self._map.name:
            # Gas station
            complex_junctions.append([
                self._map.get_waypoint_xodr(518, -1, 8.1).get_junction(),
                self._map.get_waypoint_xodr(886, 1, 10.11).get_junction(),
                self._map.get_waypoint_xodr(467, 1, 25.8).get_junction()])

        self._fake_lane_pair_keys.extend(fake_lane_keys)
        return complex_junctions

    def _join_complex_junctions(self, filtered_data):
        """
        Joins complex junctions into one. This makes it such that all the junctions,
        as well as their connecting lanes, are treated as the same junction
        """
        route_data = []
        prev_index = -1

        # If entering a complex, add all its junctions to the list
        for junction_data in filtered_data:
            junction = junction_data.junctions[0]
            prev_junction = route_data[-1] if len(route_data) > 0 else None

            # Get the complex index
            current_index = -1
            for i, complex_junctions in enumerate(self._get_complex_junctions()):
                complex_ids = [j.id for j in complex_junctions]
                if junction.id in complex_ids:
                    current_index = i
                    break

            if current_index == -1:
                # Outside a complex, add it
                route_data.append(junction_data)

            elif current_index == prev_index:
                # Same complex as the previous junction
                prev_junction.route_exit_index = junction_data.route_exit_index

            else:
                # New complex, add it
                junction_ids = [j.id for j in junction_data.junctions]
                for complex_junction in complex_junctions:
                    if complex_junction.id not in junction_ids:
                        junction_data.junctions.append(complex_junction)

                route_data.append(junction_data)

            prev_index = current_index

        return route_data

    def _get_fake_lane_pairs(self, fake_data):
        """Gets a list of entry-exit lanes of the fake junctions"""
        for fake_junctions_data in fake_data:
            for junction in fake_junctions_data.junctions:
                for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
                    while self._is_junction(entry_wp):
                        entry_wps = entry_wp.previous(0.5)
                        if len(entry_wps) == 0:
                            break  # Stop when there's no prev
                        entry_wp = entry_wps[0]
                    if self._is_junction(entry_wp):
                        continue  # Triggered by the loops break

                    while self._is_junction(exit_wp):
                        exit_wps = exit_wp.next(0.5)
                        if len(exit_wps) == 0:
                            break  # Stop when there's no prev
                        exit_wp = exit_wps[0]
                    if self._is_junction(exit_wp):
                        continue  # Triggered by the loops break

                    self._fake_junction_ids.append(junction.id)
                    self._fake_lane_pair_keys.append([get_lane_key(entry_wp), get_lane_key(exit_wp)])

    def _get_junction_entry_wp(self, entry_wp):
        """For a junction waypoint, returns a waypoint outside of it that entrys into its lane"""
        # Exit the junction
        while self._is_junction(entry_wp):
            entry_wps = entry_wp.previous(0.2)
            if len(entry_wps) == 0:
                return None  # Stop when there's no prev
            entry_wp = entry_wps[0]
        return entry_wp

    def _get_junction_exit_wp(self, exit_wp):
        """For a junction waypoint, returns a waypoint outside of it from which the lane exits the junction"""
        while self._is_junction(exit_wp):
            exit_wps = exit_wp.next(0.2)
            if len(exit_wps) == 0:
                return None  # Stop when there's no prev
            exit_wp = exit_wps[0]
        return exit_wp

    def _get_closest_junction_waypoint(self, wp, junction_wps):
        """matches a given wp to another one inside the list. This is first donw by checking its key,
        and if this fails, the closest wp is chosen"""

        # Check the lane keys
        junction_keys = [get_lane_key(wp_) for wp_ in junction_wps]
        if get_lane_key(wp) in junction_keys:
            return wp

        # Get the closest one
        closest_dist = float('inf')
        junction_wp = None
        route_location = wp.transform.location
        for wp_ in junction_wps:
            distance = wp_.transform.location.distance(route_location)
            if distance < closest_dist:
                closest_dist = distance
                junction_wp = wp_

        return junction_wp

    def _is_route_wp_behind_junction_wp(self, route_wp, junction_wp):
        """Checks if an actor is behind the ego. Uses the route transform"""
        route_location = route_wp.transform.location
        junction_transform = junction_wp.transform
        junction_heading = junction_transform.get_forward_vector()
        wps_vec = route_location - junction_transform.location
        if junction_heading.x * wps_vec.x + junction_heading.y * wps_vec.y < - 0.09: # 85º
            return True
        return False

    def _add_junctions_topology(self, route_data):
        """Gets the entering and exiting lanes of a multijunction"""
        for junction_data in route_data:
            used_entry_lanes = []
            used_exit_lanes = []
            entry_lane_wps = []
            exit_lane_wps = []

            if self.debug:
                print(' --------------------- ')
            for junction in junction_data.junctions:
                for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving): 

                    entry_wp = self._get_junction_entry_wp(entry_wp)
                    if not entry_wp:
                        continue
                    if get_lane_key(entry_wp) not in used_entry_lanes:
                        used_entry_lanes.append(get_lane_key(entry_wp))
                        entry_lane_wps.append(entry_wp)
                        if self.debug:
                            draw_point(self._world, entry_wp.transform.location, 'small', 'entry', True)

                    exit_wp = self._get_junction_exit_wp(exit_wp)
                    if not exit_wp:
                        continue
                    if get_lane_key(exit_wp) not in used_exit_lanes:
                        used_exit_lanes.append(get_lane_key(exit_wp))
                        exit_lane_wps.append(exit_wp)
                        if self.debug:
                            draw_point(self._world, exit_wp.transform.location, 'small', 'exit', True)

            # Check for connecting lanes. This is pretty much for the roundabouts, but some weird geometries
            # make it possible for single junctions to have the same road entering and exiting. Two cases,
            # Lanes that exit one junction and enter another (or viceversa)
            exit_lane_keys = [get_lane_key(wp) for wp in exit_lane_wps]
            entry_lane_keys = [get_lane_key(wp) for wp in entry_lane_wps]
            for wp in list(entry_lane_wps):
                if get_lane_key(wp) in exit_lane_keys:
                    entry_lane_wps.remove(wp)
                    if self.debug:
                        draw_point(self._world, wp.transform.location, 'small', 'connect', True)

            for wp in list(exit_lane_wps):
                if get_lane_key(wp) in entry_lane_keys:
                    exit_lane_wps.remove(wp)
                    if self.debug:
                        draw_point(self._world, wp.transform.location, 'small', 'connect', True)

            # Lanes with a fake junction in the middle (maps junction exit to fake junction entry and viceversa)
            for entry_key, exit_key in self._fake_lane_pair_keys:
                entry_wp = None
                for wp in entry_lane_wps:
                    if get_lane_key(wp) == exit_key:  # A junction exit is a fake junction entry
                        entry_wp = wp
                        break
                exit_wp = None
                for wp in exit_lane_wps:
                    if get_lane_key(wp) == entry_key:  # A junction entry is a fake junction exit
                        exit_wp = wp
                        break
                if entry_wp and exit_wp:
                    entry_lane_wps.remove(entry_wp)
                    exit_lane_wps.remove(exit_wp)
                    if self.debug:
                        draw_point(self._world, entry_wp.transform.location, 'small', 'connect', True)
                        draw_point(self._world, exit_wp.transform.location, 'small', 'connect', True)

            junction_data.entry_wps = entry_lane_wps
            junction_data.exit_wps = exit_lane_wps

            # Filter the entries and exits that correspond to the route
            route_entry_wp = self._route[junction_data.route_entry_index]

            # Junction entry
            for wp in get_same_dir_lanes(route_entry_wp):
                junction_wp = self._get_closest_junction_waypoint(wp, entry_lane_wps)
                junction_data.route_entry_keys.append(get_lane_key(junction_wp))
            for wp in get_opposite_dir_lanes(route_entry_wp):
                junction_wp = self._get_closest_junction_waypoint(wp, exit_lane_wps)
                junction_data.opposite_exit_keys.append(get_lane_key(junction_wp))

            # Junction exit
            if junction_data.route_exit_index:  # Can be None if route ends at a junction
                route_exit_wp = self._route[junction_data.route_exit_index]
                for wp in get_same_dir_lanes(route_exit_wp):
                    junction_wp = self._get_closest_junction_waypoint(wp, exit_lane_wps)
                    junction_data.route_exit_keys.append(get_lane_key(junction_wp))
                for wp in get_opposite_dir_lanes(route_exit_wp):
                    junction_wp = self._get_closest_junction_waypoint(wp, entry_lane_wps)
                    junction_data.opposite_entry_keys.append(get_lane_key(junction_wp))

            # Add the entry directions of each lane with respect to the route. Used for scenarios 7 to 9
            route_entry_yaw = route_entry_wp.transform.rotation.yaw
            for wp in entry_lane_wps:
                diff = (wp.transform.rotation.yaw - route_entry_yaw) % 360
                if diff > 330.0:
                    direction = 'ref'
                elif diff > 225.0:
                    direction = 'right'
                elif diff > 135.0:
                    direction = 'opposite'
                elif diff > 30.0:
                    direction = 'left'
                else:
                    direction = 'ref'

                junction_data.entry_directions[direction].append(get_lane_key(wp))

            # Supposing scenario vehicles go straight, these correspond to the exit lanes of the entry directions
            for wp in exit_lane_wps:
                diff = (wp.transform.rotation.yaw - route_entry_yaw) % 360
                if diff > 330.0:
                    direction = 'ref'
                elif diff > 225.0:
                    direction = 'right'
                elif diff > 135.0:
                    direction = 'opposite'
                elif diff > 30.0:
                    direction = 'left'
                else:
                    direction = 'ref'

                junction_data.exit_directions[direction].append(get_lane_key(wp))

            if self.debug:
                exit_lane = self._route[junction_data.route_exit_index] if junction_data.route_exit_index else None
                print('> R Entry Lane: {}'.format(get_lane_key(self._route[junction_data.route_entry_index])))
                print('> R Exit  Lane: {}'.format(get_lane_key(exit_lane)))
                entry = '> J Entry Lanes: '
                for entry_wp in entry_lane_wps:
                    key = get_lane_key(entry_wp)
                    entry += key + ' ' * (6 - len(key))
                print(entry)
                exit = '> J Exit  Lanes: '
                for exit_wp in exit_lane_wps:
                    key = get_lane_key(exit_wp)
                    exit += key + ' ' * (6 - len(key))
                print(exit)
                route_entry = '> R-J Entry Lanes: '
                for entry_key in junction_data.route_entry_keys:
                    route_entry += entry_key + ' ' * (6 - len(entry_key))
                print(route_entry)
                route_exit = '> R-J Route Exit  Lanes: '
                for exit_key in junction_data.route_exit_keys:
                    route_exit += exit_key + ' ' * (6 - len(exit_key))
                print(route_exit)
                route_oppo_entry = '> R-J Oppo Entry Lanes: '
                for oppo_entry_key in junction_data.opposite_entry_keys:
                    route_oppo_entry += oppo_entry_key + ' ' * (6 - len(oppo_entry_key))
                print(route_oppo_entry)
                route_oppo_exit = '> R-J Oppo Exit  Lanes: '
                for oppo_exit_key in junction_data.opposite_exit_keys:
                    route_oppo_exit += oppo_exit_key + ' ' * (6 - len(oppo_exit_key))
                print(route_oppo_exit)

    ################################
    ## Waypoint related functions ##
    ################################

    def _is_junction(self, wp):
        if not wp.is_junction or wp.junction_id in self._fake_junction_ids:
            return False
        return True

    ################################
    ##       Mode functions       ##
    ################################

    def _add_actor_dict_element(self, actor_dict, actor, exit_lane_key='', at_oppo_entry_lane=False):
        """Adds a new actor to the actor dictionary"""
        actor_dict[actor] = {
            'state': 'junction_entry' if not exit_lane_key else 'junction_exit',
            'exit_lane_key': exit_lane_key,
            'at_oppo_entry_lane': at_oppo_entry_lane
        }

    def _switch_to_junction_mode(self, junction):
        """Prepares the junction mode, changing the state of the actors"""
        self._ego_state = 'junction'
        for actor in list(self._road_actors):
            self._add_actor_dict_element(junction.actor_dict, actor)
            self._road_actors.remove(actor)
            if not self._is_scenario_2_active:
                self._tm.vehicle_percentage_speed_difference(actor, 0)

        self._opposite_sources.clear()

    def _initialise_junction_scenario(self, direction, remove_entries, remove_exits, remove_middle):
        """
        Removes all vehicles in a particular 'direction' as well as all actors inside the junction.
        Additionally, activates some flags to ensure the junction is empty at all times
        """
        self._direction = direction.lower()
        self._remove_entries = remove_entries
        self._remove_exits = remove_exits
        self._remove_middle = remove_middle

        if self._active_junctions:
            scenario_junction = self._active_junctions[0]
            entry_direction_keys = scenario_junction.entry_directions[direction]
            actor_dict = scenario_junction.actor_dict

            if self._remove_entries:
                for entry_source in scenario_junction.entry_sources:
                    if get_lane_key(entry_source.entry_lane_wp) in entry_direction_keys:
                        # Source is affected
                        actors = entry_source.actors
                        for actor in list(actors):
                            if actor_dict[actor]['state'] == 'junction_entry':
                                # Actor is at the entry lane
                                self._destroy_actor(actor)

            if self._remove_exits:
                for exit in scenario_junction.exit_directions[direction]:
                    for actor in list(scenario_junction.exit_dict[exit]['actors']):
                        self._destroy_actor(actor)

            if self._remove_middle:
                actor_dict = scenario_junction.actor_dict
                for actor in list(actor_dict):
                    if actor_dict[actor]['state'] == 'junction_middle':
                        self._destroy_actor(actor)

    def _handle_junction_scenario_end(self):
        """Ends the junction scenario interaction"""
        self._direction = None
        self._remove_entries = None
        self._remove_exits = None
        self._remove_middle = None

    def _monitor_scenario_4_end(self, ego_location):
        """Monitors the ego distance to the junction to know if the scenario 4 has ended"""
        if self._ego_exitted_junction:
            ref_location = self._start_ego_wp.transform.location
            if ego_location.distance(ref_location) > self._crossing_dist:
                for actor in self._scenario_4_actors:
                    self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._is_scenario_4_active = False
                self._scenario_4_actors.clear()
                self._ego_exitted_junction = False
                self._crossing_dist = None

    def _handle_scenario_4_interaction(self, junction, ego_wp):
        """
        Handles the interation between the scenario 4 of the Leaderboard and the background activity.
        This removes all vehicles near the bycicle path, and stops the others so that they don't interfere
        """
        if not self._is_scenario_4_active:
            return

        self._ego_exitted_junction = True
        self._start_ego_wp = ego_wp
        min_crossing_space = 2

        # Actor exitting the junction
        exit_dict = junction.exit_dict
        for exit_key in exit_dict:
            if exit_key not in junction.route_exit_keys:
                continue
            actors = exit_dict[exit_key]['actors']
            exit_lane_wp = exit_dict[exit_key]['ref_wp']
            exit_lane_location = exit_lane_wp.transform.location
            for actor in list(actors):
                actor_location = CarlaDataProvider.get_location(actor)
                if not actor_location:
                    self._destroy_actor(actor)
                    continue

                dist_to_scenario = exit_lane_location.distance(actor_location) - self._crossing_dist
                actor_length = actor.bounding_box.extent.x
                if abs(dist_to_scenario) < actor_length + min_crossing_space:
                    self._destroy_actor(actor)
                    continue

                if dist_to_scenario > 0:
                    continue  # Don't stop the actors that have already passed the scenario

                if get_lane_key(ego_wp) == get_lane_key(exit_lane_wp):
                    self._destroy_actor(actor)
                    continue  # Actor at the ego lane and between the ego and scenario

                self._scenario_4_actors.append(actor)

        # Actor entering the junction
        for entry_source in junction.entry_sources:
            entry_lane_wp = entry_source.entry_lane_wp
            if get_lane_key(entry_lane_wp) in junction.opposite_entry_keys:
                # Source is affected
                actors = entry_source.actors
                entry_lane_location = entry_lane_wp.transform.location
                for actor in list(actors):
                    actor_location = CarlaDataProvider.get_location(actor)
                    if not actor_location:
                        self._destroy_actor(actor)
                        continue

                    crossing_space = abs(entry_lane_location.distance(actor_location) - self._crossing_dist)
                    actor_length = actor.bounding_box.extent.x
                    if crossing_space < actor_length + min_crossing_space:
                        self._destroy_actor(actor)
                        continue  # Actors blocking the path of the crossing obstacle

                    self._scenario_4_actors.append(actor)

        # Actors entering the next junction
        if len(self._active_junctions) > 1:
            next_junction = self._active_junctions[1]
            actors_dict = next_junction.actor_dict
            for actor in list(actors_dict):
                if actors_dict[actor]['state'] != 'junction_entry':
                    continue

                actor_location = CarlaDataProvider.get_location(actor)
                if not actor_location:
                    self._destroy_actor(actor)
                    continue

                dist_to_scenario = exit_lane_location.distance(actor_location) - self._crossing_dist
                actor_length = actor.bounding_box.extent.x
                if abs(dist_to_scenario) < actor_length + min_crossing_space:
                    self._destroy_actor(actor)
                    continue

                if dist_to_scenario > 0:
                    continue  # Don't stop the actors that have already passed the scenario

                actor_wp = self._map.get_waypoint(actor_location)
                if get_lane_key(ego_wp) == get_lane_key(actor_wp):
                    self._destroy_actor(actor)
                    continue  # Actor at the ego lane and between the ego and scenario

                self._scenario_4_actors.append(actor)


        # Immediately freeze the actors
        for actor in self._scenario_4_actors:
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            self._tm.vehicle_percentage_speed_difference(actor, 100)

    def _end_junction_behavior(self, ego_wp, junction):
        """
        Destroys unneeded actors (those behind the ego), moves the rest to other data structures
        and cleans up the variables. If no other junctions are active, starts road mode
        """
        actor_dict = junction.actor_dict
        route_exit_keys = junction.route_exit_keys

        active_junctions = len(self._active_junctions)

        for actor in list(actor_dict):
            location = CarlaDataProvider.get_location(actor)
            if not location or self._is_location_behind_ego(location):
                self._destroy_actor(actor)
                continue

            self._tm.vehicle_percentage_speed_difference(actor, 0)
            if actor_dict[actor]['at_oppo_entry_lane']:
                self._opposite_actors.append(actor)
                self._tm.ignore_lights_percentage(actor, 100)
                self._tm.ignore_signs_percentage(actor, 100)
                continue

            if active_junctions <= 1 and actor_dict[actor]['exit_lane_key'] in route_exit_keys:
                self._road_actors.append(actor)
                continue

            self._destroy_actor(actor)

        self._handle_scenario_4_interaction(junction, ego_wp)
        self._handle_junction_scenario_end()

        self._switch_junction_road_sources(junction)
        self._active_junctions.pop(0)

        if not self._active_junctions:
            self._ego_state = 'road'
            self._initialise_opposite_sources()
            self._initialise_road_checker()

    def _switch_junction_road_sources(self, junction):
        """
        Removes the sources part of the previous road and gets the ones of the exitted junction.
        """
        self._road_sources.clear()
        self._road_sources.extend(junction.exit_sources)

    def _search_for_next_junction(self):
        """Check if closeby to a junction. The closest one will always be the first"""
        if not self._junctions:
            return None

        ego_accum_dist = self._accum_dist[self._route_index]
        junction_accum_dist = self._accum_dist[self._junctions[0].route_entry_index]
        if junction_accum_dist - ego_accum_dist < self._junction_detection_dist:  # Junctions closeby
            return self._junctions.pop(0)

        return None

    def _initialise_connecting_lanes(self, junction):
        """
        Moves the actors currently at the exit lane of the last junction
        to entry actors of the newly created junction
        """
        if len(self._active_junctions) > 0:
            prev_junction = self._active_junctions[-1]
            route_exit_keys = prev_junction.route_exit_keys
            exit_dict = prev_junction.exit_dict
            for exit_key in route_exit_keys:
                exit_actors = exit_dict[exit_key]['actors']
                for actor in list(exit_actors):
                    self._remove_actor_info(actor)
                    self._add_actor_dict_element(junction.actor_dict, actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 0)

    def _monitor_nearby_junctions(self):
        """
        Monitors when the ego approaches a junction, preparing the junction mode when it happens.
        This can be triggered even if there is another junction behavior happening
        """
        junction = self._search_for_next_junction()
        if not junction:
            return

        if self._ego_state == 'road':
            self._switch_to_junction_mode(junction)
        self._initialise_junction_sources(junction)
        self._initialise_junction_exits(junction)
        self._initialise_connecting_lanes(junction)
        self._active_junctions.append(junction)

    def _monitor_ego_junction_exit(self, ego_wp):
        """
        Monitors when the ego exits the junctions, preparing the road mode when that happens
        """
        current_junction = self._active_junctions[0]
        exit_index = current_junction.route_exit_index
        if exit_index and self._route_index >= exit_index:
            self._end_junction_behavior(ego_wp, current_junction)

    def _add_incoming_actors(self, junction, source):
        """Checks nearby actors that will pass through the source, adding them to it"""
        source_location = source.wp.transform.location
        if not source.previous_lane_keys:
            source.previous_lane_keys = [get_lane_key(prev_wp) for prev_wp in source.wp.previous(self._reuse_dist)]
            source.previous_lane_keys.append(get_lane_key(source.wp))

        for actor in self._get_actors():
            if actor in source.actors:
                continue  # Don't use actors already part of the source

            actor_location = CarlaDataProvider.get_location(actor)
            if actor_location is None:
                continue  # No idea where the actor is, ignore it
            if source_location.distance(actor_location) > self._reuse_dist:
                continue  # Don't use actors far away

            actor_wp = self._map.get_waypoint(actor_location)
            if get_lane_key(actor_wp) not in source.previous_lane_keys:
                continue  # Don't use actors that won't pass through the source

            self._tm.vehicle_percentage_speed_difference(actor, 0)
            self._remove_actor_info(actor)
            source.actors.append(actor)

            at_oppo_entry_lane = get_lane_key(source.entry_lane_wp) in junction.opposite_entry_keys
            self._add_actor_dict_element(junction.actor_dict, actor, at_oppo_entry_lane=at_oppo_entry_lane)

            return actor

    def _update_road_sources(self, ego_location):
        """
        Manages the sources that spawn actors behind the ego.
        Sources are destroyed after their actors are spawned
        """
        for source in list(self._road_sources):
            if self.debug:
                draw_point(self._world, source.wp.transform.location, 'small', self._ego_state, False)
                draw_string(self._world, source.wp.transform.location, str(len(source.actors)), self._ego_state, False)

            if len(source.actors) >= self._road_back_vehicles:
                self._road_sources.remove(source)
                continue

            if len(source.actors) == 0:
                location = ego_location
            else:
                location = CarlaDataProvider.get_location(source.actors[-1])
                if not location:
                    continue

            distance = location.distance(source.wp.transform.location)

            # Spawn a new actor if the last one is far enough
            if distance > self._road_spawn_dist:
                actor = self._spawn_source_actor(source, ego_stop=False)
                if actor is None:
                    continue

                self._tm.distance_to_leading_vehicle(actor, self._road_vehicle_dist)
                self._tm.auto_lane_change(actor, False)
                source.actors.append(actor)
                if self._ego_state == 'road':
                    self._road_actors.append(actor)
                elif self._ego_state == 'junction':
                    self._add_actor_dict_element(self._active_junctions[0].actor_dict, actor)

    ################################
    ## Behavior related functions ##
    ################################

    def _initialise_road_behavior(self, road_wps, ego_wp):
        """Intialises the road behavior, consisting on several vehicle in front of the ego"""
        spawn_wps = []
        # Vehicles in front
        for wp in road_wps:
            next_wp = wp
            for _ in range(self._road_front_vehicles):
                next_wps = next_wp.next(self._road_spawn_dist)
                if len(next_wps) != 1 or self._is_junction(next_wps[0]):
                    break  # Stop when there's no next or found a junction
                next_wp = next_wps[0]
                spawn_wps.append(next_wp)

        # Vehicles on the side
        for wp in road_wps:
            if wp.lane_id != ego_wp.lane_id:
                spawn_wps.append(wp)

        for actor in self._spawn_actors(spawn_wps):
            self._tm.distance_to_leading_vehicle(actor, self._road_vehicle_dist)
            self._tm.auto_lane_change(actor, False)
            self._road_actors.append(actor)

        # Vehicles behind
        for wp in road_wps:
            spawn_wps = []
            prev_wp = wp
            for _ in range(self._road_back_vehicles):
                prev_wps = prev_wp.previous(self._road_spawn_dist)
                if len(prev_wps) != 1 or self._is_junction(prev_wps[0]):
                    break  # Stop when there's no next or found a junction
                prev_wp = prev_wps[0]
                spawn_wps.append(prev_wp)

            actors = self._spawn_actors(spawn_wps)
            for actor in actors:
                self._tm.distance_to_leading_vehicle(actor, self._road_vehicle_dist)
                self._tm.auto_lane_change(actor, False)
                self._road_actors.append(actor)

            if len(actors) < self._road_back_vehicles:
                self._road_sources.append(Source(prev_wp, actors))

    def _initialise_opposite_sources(self):
        """
        Gets the waypoints where the actor sources that spawn actors in the opposite direction
        will be located. These are at a fixed distance from the ego, but never entering junctions
        """
        self._opposite_route_index = None
        if not self._junctions:
            self._opposite_route_index = self._route_length - 1
        else:
            next_junction_index = self._junctions[0].route_entry_index
            ego_accum_dist = self._accum_dist[self._route_index]
            for i in range(self._route_index, next_junction_index):
                if self._accum_dist[i] - ego_accum_dist > self._opposite_sources_dist:
                    self._opposite_route_index = i
                    break
            if not self._opposite_route_index:
                # Junction is closer than the opposite source distance
                self._opposite_route_index = next_junction_index

        oppo_wp = self._route[self._opposite_route_index]

        for wp in get_opposite_dir_lanes(oppo_wp):
            self._opposite_sources.append(Source(wp, []))

    def _initialise_road_checker(self):
        """
        Gets the waypoints in front of the ego to continuously check if the road changes
        """
        self._road_checker_index = None

        if not self._junctions:
            upper_limit = self._route_length - 1
        else:
            upper_limit = self._junctions[0].route_entry_index

        ego_accum_dist = self._accum_dist[self._route_index]
        for i in range(self._route_index, upper_limit):
            if self._accum_dist[i] - ego_accum_dist > self._max_radius:
                self._road_checker_index = i
                break
        if not self._road_checker_index:
            self._road_checker_index = upper_limit

    def _initialise_junction_sources(self, junction):
        """
        Initializes the actor sources to ensure the junction is always populated. They are 
        placed at certain distance from the junction, but are stopped if another junction is found,
        to ensure the spawned actors always move towards the activated one
        """
        for wp in junction.entry_wps:
            entry_lane_key = get_lane_key(wp)
            if entry_lane_key in junction.route_entry_keys:
                continue  # Ignore the road from which the route enters

            if self._remove_entries and entry_lane_key in junction.entry_directions[self._direction]:
                continue  # Ignore entries that are part of active junction scenarios

            moved_dist = 0
            prev_wp = wp
            while moved_dist < self._junction_sources_dist:
                prev_wps = prev_wp.previous(5)
                if len(prev_wps) == 0 or self._is_junction(prev_wps[0]):
                    break
                prev_wp = prev_wps[0]
                moved_dist += 5

            junction.entry_sources.append(Source(prev_wp, [], entry_lane_wp=wp))

    def _initialise_junction_exits(self, junction):
        """
        Computes and stores the max capacity of the exit. Prepares the behavior of the next road
        by creating actors at the route exit, and the sources that'll create actors behind the ego
        """
        exit_wps = junction.exit_wps
        route_exit_keys = junction.route_exit_keys

        for wp in exit_wps:
            max_actors = 0
            max_distance = 0
            exiting_wps = []

            next_wp = wp
            for i in range(max(self._road_front_vehicles, 1)):

                # Get the moving distance (first jump is higher to allow space for another vehicle)
                if i == 0:
                    move_dist = 2 * self._junction_spawn_dist
                else:
                    move_dist = self._junction_spawn_dist

                # And move such distance
                next_wps = next_wp.next(move_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                if max_actors > 0 and self._is_junction(next_wp):
                    break  # Stop when a junction is found

                max_actors += 1
                max_distance += move_dist
                exiting_wps.insert(0, next_wp)

            junction.exit_dict[get_lane_key(wp)] = {
                'actors': [], 'max_actors': max_actors, 'ref_wp': wp, 'max_distance': max_distance
            }

            exit_lane_key = get_lane_key(wp)
            if self._remove_exits and exit_lane_key in junction.exit_directions[self._direction]:
                continue  # The direction is prohibited as a junction scenario is active

            if exit_lane_key in route_exit_keys:
                junction.exit_sources.append(Source(wp, []))

                actors = self._spawn_actors(exiting_wps)
                for actor in actors:
                    self._tm.distance_to_leading_vehicle(actor, self._junction_vehicle_dist)
                    self._tm.auto_lane_change(actor, False)
                    self._add_actor_dict_element(junction.actor_dict, actor, exit_lane_key=exit_lane_key)
                junction.exit_dict[exit_lane_key]['actors'] = actors

    def _update_junction_sources(self):
        """Checks the actor sources to see if new actors have to be created"""
        for junction in self._active_junctions:
            actor_dict = junction.actor_dict
            for source in junction.entry_sources:
                if self.debug:
                    draw_point(self._world, source.wp.transform.location, 'small', 'junction', False)
                    draw_string(self._world, source.wp.transform.location, str(len(source.actors)), 'junction', False)

                entry_lane_key = get_lane_key(source.entry_lane_wp)
                at_oppo_entry_lane = entry_lane_key in junction.opposite_entry_keys

                # The direction is prohibited as a junction scenario is active
                if self._remove_entries and entry_lane_key in junction.entry_directions[self._direction]:
                    continue

                self._add_incoming_actors(junction, source)

                # Cap the amount of alive actors
                if len(source.actors) >= self._junction_sources_max_actors:
                    continue

                # Calculate distance to the last created actor
                if len(source.actors) == 0:
                    distance = self._junction_spawn_dist + 1
                else:
                    actor_location = CarlaDataProvider.get_location(source.actors[-1])
                    if not actor_location:
                        continue
                    distance = actor_location.distance(source.wp.transform.location)

                # Spawn a new actor if the last one is far enough
                if distance > self._junction_spawn_dist:
                    actor = self._spawn_source_actor(source)
                    if not actor:
                        continue

                    self._tm.distance_to_leading_vehicle(actor, self._junction_vehicle_dist)
                    self._tm.auto_lane_change(actor, False)
                    self._add_actor_dict_element(actor_dict, actor, at_oppo_entry_lane=at_oppo_entry_lane)
                    source.actors.append(actor)

    def _found_a_road_change(self, old_index, new_index):
        """Checks if the new route waypoint is part of a new road (excluding fake junctions)"""
        if new_index == old_index:
            return False

        new_wp = self._route[new_index]
        old_wp = self._route[old_index]
        if get_road_key(new_wp) == get_road_key(old_wp):
            return False

        new_wp_junction = new_wp.get_junction()
        if new_wp_junction and new_wp_junction.id in self._fake_junction_ids:
            return False

        return True

    def _move_road_checker(self, prev_index, current_index):
        """
        Continually check the road in front to see if it has changed its topology.
        If so and the number of lanes have reduced, remove the actor of the lane that merges into others
        """
        if self.debug:
            checker_wp = self._route[self._road_checker_index]
            draw_point(self._world, checker_wp.transform.location, 'small', 'road', False)

        if prev_index == current_index:
            return

        # Get the new route tracking wp
        checker_index = None
        last_index = self._junctions[0].route_entry_index if self._junctions else self._route_length - 1
        current_accum_dist = self._accum_dist[current_index]
        for i in range(self._road_checker_index, last_index):
            accum_dist = self._accum_dist[i]
            if accum_dist - current_accum_dist >= self._max_radius:
                checker_index = i
                break
        if not checker_index:
            checker_index = last_index

        if self._found_a_road_change(self._road_checker_index, checker_index):
            # Do something
            new_wps = get_same_dir_lanes(self._route[checker_index])
            old_wps = get_same_dir_lanes(self._route[self._road_checker_index])

            if len(new_wps) >= len(old_wps):
                pass 
            else:
                new_accum_dist = self._accum_dist[checker_index]
                prev_accum_dist = self._accum_dist[self._road_checker_index]
                route_move_dist = new_accum_dist - prev_accum_dist
                unmapped_lane_keys = []

                for old_wp in list(old_wps):
                    location = old_wp.transform.location
                    mapped_wp = None
                    for new_wp in new_wps:
                        if location.distance(new_wp.transform.location ) < 1.1 * route_move_dist:
                            mapped_wp = new_wp
                            break

                    if not mapped_wp:
                        unmapped_lane_keys.append(get_lane_key(old_wp))

                for actor in list(self._road_actors):
                    location = CarlaDataProvider.get_location(actor)
                    if not location:
                        continue
                    wp = self._map.get_waypoint(location)
                    if get_lane_key(wp) in unmapped_lane_keys:
                        self._destroy_actor(actor)

        self._road_checker_index = checker_index

    def _move_opposite_sources(self, prev_index, current_index):
        """
        Moves the sources of the opposite direction back. Additionally, tracks a point a certain distance
        in front of the ego to see if the road topology has to be recalculated
        """
        if self.debug:
            for source in self._opposite_sources:
                draw_point(self._world, source.wp.transform.location, 'small', 'opposite', False)
                draw_string(self._world, source.wp.transform.location, str(len(source.actors)), 'opposite', False)
            route_wp = self._route[self._opposite_route_index]
            draw_point(self._world, route_wp.transform.location, 'small', 'opposite', False)

        if prev_index == current_index:
            return

        # Get the new route tracking wp
        oppo_route_index = None
        last_index = self._junctions[0].route_entry_index if self._junctions else self._route_length - 1
        current_accum_dist = self._accum_dist[current_index]
        for i in range(self._opposite_route_index, last_index):
            accum_dist = self._accum_dist[i]
            if accum_dist - current_accum_dist >= self._opposite_sources_dist:
                oppo_route_index = i
                break
        if not oppo_route_index:
            oppo_route_index = last_index

        if self._found_a_road_change(self._opposite_route_index, oppo_route_index):
            # Recheck the left lanes as the topology might have changed
            new_opposite_sources = []
            new_opposite_wps = get_opposite_dir_lanes(self._route[oppo_route_index])

            # Map the old sources to the new wps, and add new ones / remove uneeded ones
            new_accum_dist = self._accum_dist[oppo_route_index]
            prev_accum_dist = self._accum_dist[self._opposite_route_index]
            route_move_dist = new_accum_dist - prev_accum_dist
            for wp in new_opposite_wps:
                location = wp.transform.location
                new_source = None
                for source in self._opposite_sources:
                    if location.distance(source.wp.transform.location ) < 1.1 * route_move_dist:
                        new_source = source
                        break

                if new_source:
                    new_source.wp = wp
                    new_opposite_sources.append(source)
                    self._opposite_sources.remove(source)
                else:
                    new_opposite_sources.append(Source(wp, []))

            self._opposite_sources = new_opposite_sources
        else:
            prev_accum_dist = self._accum_dist[prev_index]
            current_accum_dist = self._accum_dist[current_index]
            move_dist = current_accum_dist - prev_accum_dist
            if move_dist <= 0:
                return

            for source in self._opposite_sources:
                wp = source.wp
                if not self._is_junction(wp):
                    prev_wps = wp.previous(move_dist)
                    if len(prev_wps) == 0:
                        continue
                    prev_wp = prev_wps[0]
                    source.wp = prev_wp

        self._opposite_route_index = oppo_route_index

    def _update_opposite_sources(self):
        """Checks the opposite actor sources to see if new actors have to be created"""
        for source in self._opposite_sources:
            # Cap the amount of alive actors
            if len(source.actors) >= self._opposite_sources_max_actors:
                continue

            # Calculate distance to the last created actor
            if len(source.actors) == 0:
                distance = self._opposite_spawn_dist + 1
            else:
                actor_location = CarlaDataProvider.get_location(source.actors[-1])
                if not actor_location:
                    continue
                distance = source.wp.transform.location.distance(actor_location)

            # Spawn a new actor if the last one is far enough
            if distance > self._opposite_spawn_dist:
                actor = self._spawn_source_actor(source)
                if actor is None:
                    continue

                self._tm.distance_to_leading_vehicle(actor, self._opposite_vehicle_dist)
                self._tm.auto_lane_change(actor, False)
                self._opposite_actors.append(actor)
                source.actors.append(actor)

    def _update_parameters(self):
        """Changes the parameters depending on the blackboard variables and / or the speed of the ego"""
        road_behavior_data = py_trees.blackboard.Blackboard().get("BA_RoadBehavior")
        if road_behavior_data:
            num_front_vehicles, num_back_vehicles, vehicle_dist, spawn_dist = road_behavior_data
            if num_front_vehicles:
                self._road_front_vehicles = num_front_vehicles
            if num_back_vehicles:
                self._road_back_vehicles = num_back_vehicles
            if vehicle_dist:
                self._road_vehicle_dist = vehicle_dist
            if spawn_dist:
                self._road_spawn_dist = spawn_dist
            self._get_road_radius()
            py_trees.blackboard.Blackboard().set("BA_RoadBehavior", None, True)

        opposite_behavior_data = py_trees.blackboard.Blackboard().get("BA_OppositeBehavior")
        if opposite_behavior_data:
            source_dist, vehicle_dist, spawn_dist, max_actors = road_behavior_data
            if source_dist:
                if source_dist < self._junction_sources_dist:
                    print("WARNING: Opposite sources distance is lower than the junction ones. Ignoring it")
                else:
                    self._opposite_sources_dist = source_dist
            if vehicle_dist:
                self._opposite_vehicle_dist = vehicle_dist
            if spawn_dist:
                self._opposite_spawn_dist = spawn_dist
            if max_actors:
                self._opposite_sources_max_actors = max_actors
            py_trees.blackboard.Blackboard().set("BA_OppositeBehavior", None, True)

        junction_behavior_data = py_trees.blackboard.Blackboard().get("BA_JunctionBehavior")
        if junction_behavior_data:
            source_dist, vehicle_dist, spawn_dist, max_actors = road_behavior_data
            if source_dist:
                if source_dist > self._opposite_sources_dist:
                    print("WARNING: Junction sources distance is higher than the opposite ones. Ignoring it")
                else:
                    self._junction_sources_dist = source_dist
            if vehicle_dist:
                self._junction_vehicle_dist = vehicle_dist
            if spawn_dist:
                self._junction_spawn_dist = spawn_dist
            if max_actors:
                self._junction_sources_max_actors = max_actors
            py_trees.blackboard.Blackboard().set("BA_JunctionBehavior", None, True)

        break_duration = py_trees.blackboard.Blackboard().get("BA_Scenario2")
        if break_duration:
            if self._is_scenario_2_active:
                print("WARNING: A break scenario was requested but another one is already being triggered.")
            else:
                self._activate_break_scenario = True
                self._break_duration = break_duration
            py_trees.blackboard.Blackboard().set("BA_Scenario2", None, True)

        crossing_dist = py_trees.blackboard.Blackboard().get("BA_Scenario4")
        if crossing_dist:
            self._is_scenario_4_active = True
            self._crossing_dist = crossing_dist
            py_trees.blackboard.Blackboard().set("BA_Scenario4", None, True)

        direction = py_trees.blackboard.Blackboard().get("BA_Scenario7")
        if direction:
            self._initialise_junction_scenario(direction, True, True, True)
            py_trees.blackboard.Blackboard().set("BA_Scenario7", None, True)
        direction = py_trees.blackboard.Blackboard().get("BA_Scenario8")
        if direction:
            self._initialise_junction_scenario(direction, True, True, True)
            py_trees.blackboard.Blackboard().set("BA_Scenario8", None, True)
        direction = py_trees.blackboard.Blackboard().get("BA_Scenario9")
        if direction:
            self._initialise_junction_scenario(direction, True, False, True)
            py_trees.blackboard.Blackboard().set("BA_Scenario9", None, True)
        direction = py_trees.blackboard.Blackboard().get("BA_Scenario10")
        if direction:
            self._initialise_junction_scenario(direction, False, False, False)
            py_trees.blackboard.Blackboard().set("BA_Scenario10", None, True)

        speed = CarlaDataProvider.get_velocity(self._ego_actor)
        self._min_radius = self._base_min_radius + self._radius_increase_ratio*speed
        self._max_radius = self._base_max_radius + self._radius_increase_ratio*speed
        self._junction_detection_dist = self._max_radius

    def _manage_break_scenario(self):
        """
        Manages the break scenario, where all road vehicles in front of the ego suddenly stop,
        wait for a bit, and start moving again. This will never trigger unless done so from outside.
        """
        if self._is_scenario_2_active:
            self._next_scenario_time -= self._world.get_snapshot().timestamp.delta_seconds
            if self._next_scenario_time <= 0:
                for actor in self._scenario_2_actors:
                    self._tm.vehicle_percentage_speed_difference(actor, 0)
                    lights = actor.get_light_state()
                    lights &= ~carla.VehicleLightState.Brake
                    actor.set_light_state(carla.VehicleLightState(lights))
                self._scenario_2_actors = []

                self._is_scenario_2_active = False

        elif self._activate_break_scenario:
            for actor in self._road_actors:
                location = CarlaDataProvider.get_location(actor)
                if location and not self._is_location_behind_ego(location):
                    self._scenario_2_actors.append(actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 100)
                    lights = actor.get_light_state()
                    lights |= carla.VehicleLightState.Brake
                    actor.set_light_state(carla.VehicleLightState(lights))

            self._is_scenario_2_active = True
            self._activate_break_scenario = False
            self._next_scenario_time = self._break_duration

    #############################
    ##     Actor functions     ##
    #############################

    def _spawn_actors(self, spawn_wp):
        """Spawns several actors in batch"""
        spawn_transforms = []
        for wp in spawn_wp:
            spawn_transforms.append(
                carla.Transform(wp.transform.location + carla.Location(z=self._spawn_vertical_shift),
                wp.transform.rotation)
            )

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_transforms), spawn_transforms, True, False, 'background',
            safe_blueprint=True, tick=False)

        if self._night_mode:
            for actor in actors:
                actor.set_light_state(
                    carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        return actors

    def _spawn_source_actor(self, source, ego_stop=True):
        """Given a source, spawns an actor at that source"""
        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        source_transform = source.wp.transform
        if ego_stop and ego_location.distance(source_transform.location) < self._spawn_free_radius:
            return None

        new_transform = carla.Transform(
            source_transform.location + carla.Location(z=self._spawn_vertical_shift),
            source_transform.rotation
        )
        actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', new_transform, rolename='background',
            autopilot=True, random_location=False, safe_blueprint=True, tick=False)

        if self._night_mode:
            actor.set_light_state(
                carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        return actor

    def _is_location_behind_ego(self, location):
        """Checks if an actor is behind the ego. Uses the route transform"""
        ego_transform = self._route[self._route_index].transform
        ego_heading = ego_transform.get_forward_vector()
        ego_actor_vec = location - ego_transform.location
        if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < - 0.17:  # 100º
            return True
        return False

    def _update_road_actors(self, route_transform):
        """
        Dynamically controls the actor speed in front of the ego.
        Not applied to those behind it so that they can catch up it
        """
        scenario_actors = self._scenario_4_actors + self._scenario_2_actors
        for actor in self._road_actors:
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            if self.debug:
                draw_string(self._world, location, 'R', 'road', False)
            if actor in scenario_actors:
                continue
            if self._is_location_behind_ego(location):
                continue

            distance = location.distance(route_transform.location)
            speed_red = (distance - self._min_radius) / (self._max_radius - self._min_radius) * 100
            speed_red = np.clip(speed_red, 0, 100)
            self._tm.vehicle_percentage_speed_difference(actor, speed_red)

    def _update_junction_actors(self):
        """
        Handles an actor depending on their previous state. Actors entering the junction have its exit
        monitored through their waypoint. When they exit, they are either moved to a connecting junction,
        or added to the exit dictionary. Actors that exited the junction will stop after a certain distance
        """
        if len(self._active_junctions) == 0:
            return

        max_index = len(self._active_junctions) - 1
        for i, junction in enumerate(self._active_junctions):
            if self.debug:
                route_keys = junction.route_entry_keys + junction.route_exit_keys
                route_oppo_keys = junction.opposite_entry_keys + junction.opposite_exit_keys
                for wp in junction.entry_wps + junction.exit_wps:
                    if get_lane_key(wp) in route_keys:
                        draw_point(self._world, wp.transform.location, 'medium', 'road', False)
                    elif get_lane_key(wp) in route_oppo_keys:
                        draw_point(self._world, wp.transform.location, 'medium', 'opposite', False)
                    else:
                        draw_point(self._world, wp.transform.location, 'medium', 'junction', False)

            actor_dict = junction.actor_dict
            exit_dict = junction.exit_dict
            for actor in list(actor_dict):
                if actor not in actor_dict:
                    continue  # Actor was removed during the loop
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue

                state, exit_lane_key, _ = actor_dict[actor].values()
                if self.debug:
                    string = 'J' + str(i+1) + "_" + state[9:11]
                    draw_string(self._world, location, string, self._ego_state, False)

                # Monitor its entry
                if state == 'junction_entry':
                    actor_wp = self._map.get_waypoint(location)
                    if self._is_junction(actor_wp) and junction.contains(actor_wp.get_junction()):
                        if self._remove_middle:
                            self._destroy_actor(actor)  # Don't clutter the junction if a junction scenario is active
                            continue
                        actor_dict[actor]['state'] = 'junction_middle'

                # Monitor its exit and destroy an actor if needed
                elif state == 'junction_middle':
                    actor_wp = self._map.get_waypoint(location)
                    actor_lane_key = get_lane_key(actor_wp)
                    if not self._is_junction(actor_wp) and actor_lane_key in exit_dict:
                        if i < max_index and actor_lane_key in junction.route_exit_keys:
                            # Exited through a connecting lane in the route direction.
                            self._remove_actor_info(actor)
                            other_junction = self._active_junctions[i+1]
                            self._add_actor_dict_element(other_junction.actor_dict, actor)

                        elif i > 0 and actor_lane_key in junction.opposite_exit_keys:
                            # Exited through a connecting lane in the opposite direction.
                            # THIS SHOULD NEVER HAPPEN, an entry source should have already added it.
                            other_junction = self._active_junctions[i-1]
                            if actor not in other_junction.actor_dict:
                                self._remove_actor_info(actor)
                                self._add_actor_dict_element(other_junction.actor_dict, actor, at_oppo_entry_lane=True)

                        else:
                            # Check the lane capacity
                            exit_dict[actor_lane_key]['ref_wp'] = actor_wp
                            actor_dict[actor]['state'] = 'junction_exit'
                            actor_dict[actor]['exit_lane_key'] = actor_lane_key

                            actors = exit_dict[actor_lane_key]['actors']
                            if len(actors) > 0 and len(actors) >= exit_dict[actor_lane_key]['max_actors']:
                                self._destroy_actor(actors[0])  # This is always the front most vehicle
                            actors.append(actor)

                # Deactivate them when far from the junction
                elif state == 'junction_exit':
                    distance = location.distance(exit_dict[exit_lane_key]['ref_wp'].transform.location)
                    if distance > exit_dict[exit_lane_key]['max_distance']:
                        self._tm.vehicle_percentage_speed_difference(actor, 100)
                        actor_dict[actor]['state'] = 'junction_inactive'

                # Wait for something to happen
                elif state == 'junction_inactive':
                    pass

    def _update_opposite_actors(self, ref_transform):
        """
        Updates the opposite actors. This involves tracking their position,
        removing if too far behind the ego
        """
        max_dist = max(self._opposite_removal_dist, self._opposite_spawn_dist)
        for actor in list(self._opposite_actors):
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            if self.debug:
                draw_string(self._world, location, 'O', 'opposite', False)
            distance = location.distance(ref_transform.location)
            if distance > max_dist and self._is_location_behind_ego(location):
                self._destroy_actor(actor)

    def _remove_actor_info(self, actor):
        """Removes all the references of the actor"""
        if actor in self._road_actors:
            self._road_actors.remove(actor)
        if actor in self._opposite_actors:
            self._opposite_actors.remove(actor)

        for road_source in self._road_sources:
            if actor in road_source.actors:
                road_source.actors.remove(actor)
                break

        for opposite_source in self._opposite_sources:
            if actor in opposite_source.actors:
                opposite_source.actors.remove(actor)
                break

        for junction in self._active_junctions:
            junction.actor_dict.pop(actor, None)

            for exit_source in junction.exit_sources:
                if actor in exit_source.actors:
                    exit_source.actors.remove(actor)
                    break

            for entry_source in junction.entry_sources:
                if actor in entry_source.actors:
                    entry_source.actors.remove(actor)
                    break

            for exit_keys in junction.exit_dict:
                exit_actors = junction.exit_dict[exit_keys]['actors']
                if actor in exit_actors:
                    exit_actors.remove(actor)
                    break

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        self._remove_actor_info(actor)
        actor.destroy()

    def _update_ego_route_location(self, location):
        """Returns the closest route location to the ego"""
        for index in range(self._route_index, min(self._route_index + self._route_buffer, self._route_length)):

            route_wp = self._route[index]
            route_wp_dir = route_wp.transform.get_forward_vector()    # Waypoint's forward vector
            veh_wp_dir = location - route_wp.transform.location       # vector waypoint - vehicle
            dot_ve_wp = veh_wp_dir.x * route_wp_dir.x + veh_wp_dir.y * route_wp_dir.y + veh_wp_dir.z * route_wp_dir.z
            if dot_ve_wp > 0:
                self._route_index = index

        return self._route[self._route_index]

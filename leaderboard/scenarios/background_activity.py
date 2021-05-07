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


class BackgroundActivity(BasicScenario):

    """
    Implementation of a scenario to spawn a set of background actors,
    and to remove traffic jams in background traffic

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicle, config, route, debug_mode=False, timeout=0):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.route = route
        self.config = config
        self.debug = debug_mode
        self.timeout = timeout  # Timeout of scenario in seconds

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                                 [ego_vehicle],
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=True,
                                                 criteria_enable=True)

    def _initialize_actors(self, config):
        """
        Create the necessary initial actors
        """
        pass

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        # Check if a vehicle is further than X, destroy it if necessary and respawn it
        background_checker = py_trees.composites.Sequence()
        background_checker.add_child(BackgroundBehavior(
            self.ego_vehicle,
            self.route
        ))
        return background_checker

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
    def __init__(self, ego_actor, route, name="BackgroundBehavior"):
        """
        Setup class members
        """
        super(BackgroundBehavior, self).__init__(name)
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._tm.set_global_distance_to_leading_vehicle(5)

        # Actor variables
        self._ego_actor = ego_actor
        self._background_actors = []
        self._actor_dict = OrderedDict()  # Dict with {actor: {'state', 'ref_wp', 'at_junction'}}
        self._actor_dict[self._ego_actor] = {'state': 'road', 'ref_wp': None, 'at_junction': False}

        # Road variables (these are very much hardcoded so watch out when changing them)
        self._num_road_vehicles = 3  # Amount of vehicles in front of the ego
        self._vehicle_dist = 10  # Starting distance between spawned vehicles
        self._radius = 35  # Must be higher than nº_veh * veh_dist or the furthest vehicle will never activate 

        # Junction variables
        self._junction_dist = 40  # Distance at which junction behavior starts
        self._junction_ids = []
        self._exit_dict = OrderedDict()  # Dictionary to keep track of the amount of actors per lane
        self._road_exit_key = ""  # Key of the road through which the route exits the junction
        self._sources = []  # List of [source transform, last spawned actor]
        self._sources_dist = 30  # Distance from the actor sources to the junction
        self._prev_actors = []  # Keeps tracks of the previous junction actors that are still alive 
        self._max_sources = 0  # Maximum vehicles alive at the same per source

        # Route variables
        self._waypoints = []
        self._route_accum_dist = []
        prev_trans = None
        for trans, _ in route:
            self._waypoints.append(self._map.get_waypoint(trans.location))
            if prev_trans:
                dist = trans.location.distance(prev_trans.location)
                self._route_accum_dist.append(dist + self._route_accum_dist[-1])
            else:
                self._route_accum_dist.append(0)
            prev_trans = trans
        self._route_length = len(route)
        self._route_index = 0
        self._route_wp = self._waypoints[self._route_index]
        self._route_buffer = 3

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        ego_location = self._ego_actor.get_transform().location
        ego_wp = self._map.get_waypoint(ego_location)
        same_dir_wps = self._get_lanes(ego_wp)
        self._initialise_road_behavior(same_dir_wps)

    ################################
    ## Waypoint related functions ##
    ################################

    def _get_lanes(self, waypoint):
        """Gets all the lanes of the road the ego starts at"""
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

    def _lane_key(self, wp):
        """Returns a key corresponding to the lane"""
        return "" if wp is None else self._road_key(wp) + str(wp.lane_id)

    def _road_key(self, wp):
        """Returns a key corresponding to the road"""
        return "" if wp is None else str(wp.road_id) + "*"

    def _is_actor_at_exit_road(self, actor, ego_road):
        """Searches the exit lanes dictionary for a specific actor. Returns None if not found"""
        road = None
        for lane_key in self._exit_dict:
            if actor in self._exit_dict[lane_key]:
                road = int(lane_key.split('*')[0])
                break

        if road != ego_road:
            return False
        return True

    def _get_junctions_topology(self, junctions):
        """Gets the entering and exiting lanes of a junction. Wrapper around junction.get_waypoints
        as it returns wps inside the junction which might originate from the same lane.
        The entering route road"""
        used_entering_lanes = []
        used_exiting_lanes = []
        entering_lane_wps = []
        exiting_lane_wps = []

        for junction in junctions:
            for enter_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):

                # Entering waypoints. Move them out of the junction and save them
                while enter_wp.is_junction:
                    enter_wps = enter_wp.previous(0.5)
                    if len(enter_wps) == 0:
                        break  # Stop when there's no prev
                    enter_wp = enter_wps[0]
                if enter_wp.is_junction:
                    continue

                if self._lane_key(enter_wp) not in used_entering_lanes:
                    # The lane hasn't been used
                    used_entering_lanes.append(self._lane_key(enter_wp))
                    entering_lane_wps.append(enter_wp)
                
                # Exiting waypoints. Move them out of the junction and save them
                while exit_wp.is_junction:
                    exit_wps = exit_wp.next(0.5)
                    if len(exit_wps) == 0:
                        break  # Stop when there's no prev
                    exit_wp = exit_wps[0]
                if exit_wp.is_junction:
                    continue

                if self._lane_key(exit_wp) not in used_exiting_lanes:
                    # The lane hasn't been used
                    used_exiting_lanes.append(self._lane_key(exit_wp))
                    exiting_lane_wps.append(exit_wp)

            # self._world.debug.draw_point(junction.bounding_box.location + carla.Location(z=1), size=0.2, color=carla.Color(0,0,0), life_time=10)

        if len(junctions) > 1:
            # Filter junction entries that are another junction exits lanes
            exiting_lane_keys = [self._lane_key(wp) for wp in exiting_lane_wps]
            entering_lane_wps_ = [wp for wp in entering_lane_wps if self._lane_key(wp) not in exiting_lane_keys]

            # Filter junction exits that are another junction entries lanes
            entering_lane_keys = [self._lane_key(wp) for wp in entering_lane_wps]
            exiting_lane_wps_ = [wp for wp in exiting_lane_wps if self._lane_key(wp) not in entering_lane_keys]

            # for enter_wp in entering_lane_wps_:
            #     self._world.debug.draw_point(enter_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(255,255,0), life_time=10000)
            # for exit_wp in exiting_lane_wps_:
            #     self._world.debug.draw_point(exit_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(0,255,255), life_time=10000)
            return entering_lane_wps_, exiting_lane_wps_

        # for enter_wp in entering_lane_wps:
        #     self._world.debug.draw_point(enter_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(255,255,0), life_time=10000)
        # for exit_wp in exiting_lane_wps:
        #     self._world.debug.draw_point(exit_wp.transform.location + carla.Location(z=1), size=0.1, color=carla.Color(0,255,255), life_time=10000)
        return entering_lane_wps, exiting_lane_wps

    def _search_for_junctions(self, start_index):
        """Searches for a junction in front of the ego vehicle.
        When one is found, its entry and exit points are recorded.
        Several junctions closeby are treated as one"""
        junctions = []
        route_enter_wp = None
        route_exit_wp = None
        outside_junction = False
        connecting_length = 0
        start_accum_dist = self._route_accum_dist[start_index]

        end_search = False
        while not end_search:
            accum_dist = self._route_accum_dist[start_index]

            for i in range(start_index, self._route_length - 1):
                wp = self._waypoints[i]
                next_wp = self._waypoints[i+1]

                # If route starts at a junction, ignore it
                if not outside_junction:
                    if wp.is_junction:
                        continue
                    outside_junction = True

                # End of the route 
                if i == self._route_length - 2:
                    # print("End of route: "+str(i))
                    end_search = True

                # Searching for the junction exit
                if len(junctions) != 0 and route_exit_wp is None:
                    # print("Searching for end point: "+str(i))
                    if not next_wp.is_junction:
                        # print("Found end point: "+str(i))
                        route_exit_wp = next_wp
                        break

                # Searching for a junction
                elif next_wp.is_junction:
                    # print("Found junction: "+str(i))
                    junctions.append(next_wp.get_junction())
                    route_exit_wp = None
                    if route_enter_wp is None:
                        route_enter_wp = wp
                    continue

                # Reached maximum distance without finding a junction
                elif self._route_accum_dist[i] - accum_dist > self._junction_dist:
                    # print("Reached max distance: "+str(i))
                    end_search = True
                    break

                # else:
                #     print("Searching for a junction: "+str(i))

            if not end_search:
                start_index = i
                connecting_length = self._route_accum_dist[i] - start_accum_dist
                print(connecting_length)

        return junctions, route_enter_wp, route_exit_wp, connecting_length

    def _monitor_nearby_junctions(self):
        """Monitors when the ego approaches a junction"""
        # Search for the junction
        junctions, route_enter_wp, route_exit_wp, connecting_length = self._search_for_junctions(self._route_index)
        if not junctions:
            return

        self._junction_ids = [j.id for j in junctions]
        self._road_exit_key = self._road_key(route_exit_wp)
        self._max_sources = int(4 + connecting_length / 300)

        # Initialise junction mode.
        self._actor_dict[self._ego_actor]['state'] = 'junction'
        destroyed_actors = []
        for actor in self._background_actors:
            if actor in self._prev_actors:  # Remnants of previous junction
                self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': None, 'at_junction': False}
            else:
                self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._actor_dict[actor] = {'state': 'junction', 'ref_wp': None, 'at_junction': False}

        for actors in destroyed_actors:
            self._destroy_actor(actor)

        # # initialize the junction
        # print(" -------------------- ")
        enter_wps, exit_wps = self._get_junctions_topology(junctions)

        # print(" -------------------- ")
        # for enter_wp in enter_wps:
        #     print("FILTERED ENTER: " + self._lane_key(enter_wp))
        # for exit_wp in exit_wps:
        #     print("FILTERED EXIT: " + self._lane_key(exit_wp))

        self._initialise_junction_entrances(enter_wps, route_enter_wp)
        self._initialise_junction_exits(exit_wps, route_exit_wp)

    def _monitor_ego_junction_exit(self, ego_wp):
        """Monitors when the ego enters and exits the junctions"""
        at_junction = self._actor_dict[self._ego_actor]['at_junction']

        if not at_junction and ego_wp.is_junction:
            if ego_wp.get_junction().id == self._junction_ids[-1]:
                self._actor_dict[self._ego_actor]['at_junction'] = True

        elif at_junction and not ego_wp.is_junction:
            self._actor_dict[self._ego_actor] = {'state': 'road', 'ref_wp': None, 'at_junction': False}

            # Ideally destroy everything, but can't destroy actors in front of ego, so remember them instead
            destroyable_actors = []
            ego_road = ego_wp.road_id
            for actor in self._background_actors:
                location = CarlaDataProvider.get_location(actor)
                
                # Location unknown, destroy them just in case
                if not location:
                    destroyable_actors.append(actor)
                    continue
                
                # Behind the ego, destroy it
                if self._is_actor_behind(location, ego_wp.transform):
                    destroyable_actors.append(actor)
                    continue

                # Filter all actors outside the exit road of the ego
                if not self._is_actor_at_exit_road(actor, ego_road):
                    self._prev_actors.append(actor)

                self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': None, 'at_junction': False}
                self._tm.ignore_lights_percentage(actor, 0)

            for actor in destroyable_actors:
                self._destroy_actor(actor)

            self._junction = None
            self._road_exit_key = ""
            self._exit_dict.clear()
            self._sources.clear()
            self._junction_ids.clear()

    ################################
    ## Behavior related functions ##
    ################################

    def _initialise_road_behavior(self, road_wps):
        """Intialises the road behavior, consisting on several vehicle in front of the ego"""
        spawn_points = []
        for wp in road_wps:
            next_wp = wp
            for _ in range(self._num_road_vehicles):
                next_wps = next_wp.next(self._vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                spawn_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=0.2),
                    next_wp.transform.rotation
                ))

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_points), spawn_points, True, False, 'background', safe_blueprint=True, tick=False)

        # Add the actors the database and remove their lane changes
        for actor in actors:
            self._background_actors.append(actor)
            self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': None, 'at_junction': False}
            self._tm.auto_lane_change(actor, False)

    def _initialise_junction_entrances(self, enter_wps, route_enter_wp):
        """Manage the entrances of the junction, making sure the junction is always populated. This function
        initialises the actor sources"""
        for wp in enter_wps:
            if wp.road_id == route_enter_wp.road_id:
                continue  # Ignore the road from which the route exits

            # Get the transform of the virtual actor sources
            prev_wps = wp.previous(self._sources_dist)
            if len(prev_wps) == 0:
                break  # Stop when there's no prev
            prev_wp = prev_wps[0]
            source_transform = carla.Transform(
                prev_wp.transform.location + carla.Location(z=0.2),
                prev_wp.transform.rotation
            )
            self._sources.append([source_transform, []])

    def _initialise_junction_exits(self, exit_wps, route_exit_wp):
        """Manage the junction exits, making sure they are never crowded. This function initialises the lanes
        topology and spawns the actors at the route road to continue its road behavior after the ego exits the
        junction"""
        exit_route_key = self._road_key(route_exit_wp) if route_exit_wp else ""

        for wp in exit_wps:
            self._exit_dict[self._lane_key(wp)] = []
            if self._road_key(wp) != exit_route_key:
                continue  # Nothing to spawn

            exiting_points = []
            next_wp = wp
            for _ in range(self._num_road_vehicles):
                next_wps = next_wp.next(self._vehicle_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                exiting_points.append(carla.Transform(
                    next_wp.transform.location + carla.Location(z=0.2),
                    next_wp.transform.rotation
                ))

            actors = CarlaDataProvider.request_new_batch_actors(
                'vehicle.*', len(exiting_points), exiting_points, True, False, 'background',
                safe_blueprint=True, tick=False
            )
            for actor in actors:
                self._exit_dict[self._lane_key(wp)].append(actor)
                self._background_actors.append(actor)
                self._tm.auto_lane_change(actor, False)
                self._actor_dict[actor] = {'state': 'road_active', 'ref_wp': wp, 'at_junction': False}
                self._tm.ignore_lights_percentage(actor, 0)

    def _manage_junction_entrances(self):
        """Checks the actor sources to see if new actors have to be created"""
        # TODO: With multijunctions, this might fail (i.e. at gas station in Town03)
        # all_vehicles = self._world.get_actors.filter('vehicles.*')
        for i, [source_transform, actors] in enumerate(self._sources):

            if len(actors) >= self._max_sources:
                continue

            # Calculate distance to the last created actor
            if len(actors) == 0:
                distance = self._vehicle_dist + 1
            else:
                actor_location = CarlaDataProvider.get_location(actors[0])
                if actor_location is None:
                    continue
                distance = source_transform.location.distance(actor_location)

            # Spawn a new actor if the last one is far enough
            if distance > self._vehicle_dist:
                actor = CarlaDataProvider.request_new_actor(
                    'vehicle.*', source_transform, rolename='background',
                    autopilot=True, random_location=False, safe_blueprint=True, tick=False
                )
                if actor is None:
                    continue

                self._background_actors.append(actor)
                self._actor_dict[actor] = {'state': 'junction', 'ref_wp': None, 'at_junction': False}
                self._tm.auto_lane_change(actor, False)
                self._sources[i][1].insert(0, actor)

    #############################
    ## Actor related functions ##
    #############################
    def _is_actor_behind(self, location, ego_transform):
        """Checks if an actor is behind the ego. Uses the route transform"""
        ego_heading = ego_transform.get_forward_vector()
        ego_actor_vec = location - ego_transform.location
        if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < 0:
            return True
        return False

    def _manage_background_actor(self, actor, location, ego_state):
        """Handle the background actors depending on their previous state"""
        state, ref_wp, at_junction = self._actor_dict[actor].values()
        route_transform = self._waypoints[self._route_index].transform

        # Calculate the distance to the reference point (by default, to the route)
        if ref_wp is None:
            ref_location = route_transform.location
        else:
            ref_location = ref_wp.transform.location
        distance = location.distance(ref_location)

        # Deactivate if far from the reference point
        if state == 'road_active':
            if distance > self._radius + 0.5:
                self._tm.vehicle_percentage_speed_difference(actor, 100)
                if self._road_exit_key and self._road_key(ref_wp) == self._road_exit_key:
                    self._actor_dict[actor]['state'] = 'road_standby'
                else:
                    self._actor_dict[actor]['state'] = 'road_inactive'

        # Activate if closeby or destroy it if behind
        elif state == 'road_inactive':
            if distance < self._radius - 0.5:
                self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._actor_dict[actor]['state'] = 'road_active'
            if self._is_actor_behind(location, route_transform):
                self._destroy_actor(actor)

        # Special state for the actors that are part of the road behavior while the ego is at the junction
        # to avoid being destroyed
        elif state == 'road_standby':
            if ego_state != 'junction':
                self._actor_dict[actor]['state'] = 'road_actvive'

        # Monitor its exit
        elif state == 'junction':
            actor_wp = self._map.get_waypoint(location)
            if at_junction and not actor_wp.is_junction and self._lane_key(actor_wp) in self._exit_dict:
                # Last condition due to micro lanes
                lane_key = self._lane_key(actor_wp)
                if len(self._exit_dict[lane_key]) >= self._num_road_vehicles:
                    removed_actor = self._exit_dict[lane_key][-1]
                    self._destroy_actor(removed_actor)  # Remove last vehicle
                self._exit_dict[lane_key].insert(0, actor)  # Add one in front
                self._actor_dict[actor] = {
                    'state': 'road_active', 'ref_wp': actor_wp, 'at_junction': False
                }
                self._tm.ignore_lights_percentage(actor, 100)
            elif not at_junction and actor_wp.is_junction:
                if actor_wp.get_junction().id in self._junction_ids:
                    self._actor_dict[actor]['at_junction'] = True

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        if actor in self._background_actors:
            self._background_actors.remove(actor)
        if actor in self._prev_actors:
            self._prev_actors.remove(actor)

        self._actor_dict.pop(actor, None)

        if self._actor_dict[self._ego_actor]['state'] == 'junction':
            for lane_key in self._exit_dict:
                if actor in self._exit_dict[lane_key]:
                    self._exit_dict[lane_key].remove(actor)
                    break

            for i, [_, actors] in enumerate(self._sources):
                if actor in actors:
                    self._sources[i][1].remove(actor)
                    break

        actor.destroy()

    def _update_ego_route_location(self, location):
        """Returns the closest route location to the ego"""
        shortest_distance = float('inf')
        closest_index = -1

        for index in range(self._route_index, min(self._route_index + self._route_buffer, self._route_length)):
            ref_location = self._waypoints[index].transform.location
            dist_to_route = ref_location.distance(location)
            if dist_to_route <= shortest_distance:
                closest_index = index
                shortest_distance = dist_to_route

        if closest_index != -1:
            self._route_index = closest_index
        return self._waypoints[self._route_index]

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        # Handle actors removed by the TrafficManager
        # self._handle_tm_actor_destruction()

        # Get ego location and waypoint
        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        if ego_location is None:
            return new_status
        ego_route_wp = self._update_ego_route_location(ego_location)

        # Monitor the background activity's state
        ego_state = self._actor_dict[self._ego_actor]['state']
        if ego_state == 'road':
            self._monitor_nearby_junctions()
        elif ego_state == 'junction':
            self._monitor_ego_junction_exit(ego_route_wp)

        # Manage the actor sources
        if self._actor_dict[self._ego_actor]['state'] == 'junction':
            self._manage_junction_entrances()

        # Manage the background actors
        for actor in self._background_actors:
            location = CarlaDataProvider.get_location(actor)
            if location is None:
                continue
            self._world.debug.draw_string(location, self._actor_dict[actor]['state'] + str(self._actor_dict[actor]['at_junction']), False, carla.Color(0,0,0), 0.05)
            self._manage_background_actor(actor, location, ego_state)

        return new_status

        # TODO 1: TM removes vehicles!
        # TODO 2: Check for roundabout, fake junctions and all junctions
        # TODO 3: Road behavior

    def terminate(self, new_status):
        """
        Destroy all actors
        """
        for actor in self._background_actors:
            self._destroy_actor(actor)
        super(BackgroundBehavior, self).terminate(new_status)


    def _handle_tm_actor_destruction(self):
        """Some actors might be destroyed if they remain inactive for too long"""
        # vehicles = self._world.get_actors().filter('vehicle.*')
        # current_background = set([v for v in vehicles if v.attributes['role_name'] == "background"])
        # destroyed_actors = self._background_actors_set.difference(current_background)
        # self._background_actors_set = current_background

        ego_at_junction = self._actor_dict[self._ego_actor]['state'] == 'junction'
        for actor in destroyed_actors:
            print("TM removed an actor")
            # Remove it from actor list
            if actor in self._background_actors:
                self._background_actors.remove(actor)
            
            # From the dictionary
            self._actor_dict.pop(actor, None)

            # And from the junction dictionary, if ego is at a junction
            if ego_at_junction:
                for lane_key in self._exit_dict:
                    if actor in self._exit_dict[lane_key]:
                        self._exit_dict[lane_key].remove(actor)



import math
import numpy as np

import carla
from agents.navigation.local_planner import RoadOption


class Waypointer:
    EARTH_RADIUS_EQUA = 6378137.0  # 6371km

    def __init__(self, global_plan_gps, current_gnss, world):
        self.world=world
        self._global_plan_gps = []
        for node in global_plan_gps:
            gnss, cmd = node
            self._global_plan_gps.append(([gnss['lat'], gnss['lon'], gnss['z']], cmd))

        current_location = self.gps_to_location([current_gnss['lat'], current_gnss['lon'], current_gnss['z']])
        self.checkpoint = (current_location.x, current_location.y, RoadOption.LANEFOLLOW)

        self._traffic_light_map = dict()
        for traffic_light in world.get_actors().filter('*traffic_light*'):
            if traffic_light not in self._traffic_light_map.keys():
                self._traffic_light_map[traffic_light] = traffic_light.get_transform()
            else:
                raise KeyError(
                    "Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

        self.current_idx = -1

    def tick(self, gnss_data, imu_data):
        next_gps, _ = self._global_plan_gps[self.current_idx + 1]
        current_location = self.gps_to_location(gnss_data)

        tl_dist_to_last_wp = None
        try:
            next_tl, tl_dist_to_last_wp= self.get_next_traffic_light(current_location)
        except:
            pass

        next_vec_in_global = self.gps_to_location(next_gps) - self.gps_to_location(gnss_data)
        compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = self.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        # Fix the command given too late bug
        if tl_dist_to_last_wp and tl_dist_to_last_wp>10.0:
            command_trigger_condition = (np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 3.0 and loc_in_ev.x > 0.0)
        else:
            command_trigger_condition = (np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and  loc_in_ev.x < 0.0)

        if command_trigger_condition:
            self.current_idx += 1
        self.current_idx = min(self.current_idx, len(self._global_plan_gps) - 2)

        _, road_option_0 = self._global_plan_gps[max(0, self.current_idx)]
        gps_point, road_option_1 = self._global_plan_gps[self.current_idx + 1]

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0
        self.checkpoint = (current_location.x, current_location.y, road_option)

        return self.checkpoint

    def gps_to_location(self, gps):
        lat, lon, z = gps
        lat = float(lat)
        lon = float(lon)
        z = float(z)

        location = carla.Location(z=z)

        location.x = lon / 180.0 * (math.pi * self.EARTH_RADIUS_EQUA)

        location.y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * self.EARTH_RADIUS_EQUA

        return location

    def vec_global_to_ref(self, target_vec_in_global, ref_rot_in_global):
        """
        :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
        :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
        :return: carla.Vector3D in ref coordinate
        """
        R = self.carla_rot_to_mat(ref_rot_in_global)
        np_vec_in_global = np.array([[target_vec_in_global.x],
                                     [target_vec_in_global.y],
                                     [target_vec_in_global.z]])
        np_vec_in_ref = R.T.dot(np_vec_in_global)
        target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
        return target_vec_in_ref

    def carla_rot_to_mat(self, carla_rotation):
        """
        Transform rpy in carla.Rotation to rotation matrix in np.array

        :param carla_rotation: carla.Rotation
        :return: np.array rotation matrix
        """
        roll = np.deg2rad(carla_rotation.roll)
        pitch = np.deg2rad(carla_rotation.pitch)
        yaw = np.deg2rad(carla_rotation.yaw)

        yaw_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        pitch_matrix = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])
        roll_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
        return rotation_matrix

    def get_next_traffic_light(self, location):
        """
        returns the next relevant traffic light for the provided actor, or waypoint
        """

        waypoint = self.world.get_map().get_waypoint(location)

        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(1.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in self._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = self._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light, distance_to_relevant_traffic_light
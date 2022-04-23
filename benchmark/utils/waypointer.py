import math
import numpy as np

import carla
from agents.navigation.local_planner import RoadOption


class Waypointer:
    EARTH_RADIUS_EQUA = 6378137.0  # 6371km

    def __init__(self, global_plan_gps, current_gnss):
        self._global_plan_gps = []
        for node in global_plan_gps:
            gnss, cmd = node
            self._global_plan_gps.append(([gnss['lat'], gnss['lon'], gnss['z']], cmd))

        current_location = self.gps_to_location(current_gnss)
        self.checkpoint = (current_location.x, current_location.y, RoadOption.LANEFOLLOW)

        self.current_idx = -1

    def tick(self, gnss_data, imu_data):

        next_gps, _ = self._global_plan_gps[self.current_idx + 1]
        current_location = self.gps_to_location(gnss_data)

        next_vec_in_global = self.gps_to_location(next_gps) - self.gps_to_location(gnss_data)
        compass = 0.0 if np.isnan(imu_data[-1]) else imu_data[-1]
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = self.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)
        if np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and loc_in_ev.x < 3.0:
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
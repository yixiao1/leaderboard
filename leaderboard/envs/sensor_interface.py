import copy
import logging
import numpy as np
import os
import time
from threading import Thread

from queue import Queue
from queue import Empty
from threading import Lock

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.waypointer import Waypointer

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) or first_time or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {'speed': self._get_forward_speed(transform=transform, velocity=velocity)}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}

class CanbusReader(BaseReader):
    """
    Sensor to measure the action of the vehicle.
    """
    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """
        control = self._vehicle.get_control()
        transform = self._vehicle.get_transform()
        ego_waypoint = CarlaDataProvider.get_map().get_waypoint(self._vehicle.get_location())
        return {
            'ego_position': [transform.location.x, transform.location.y, transform.location.z],
            'steer': np.nan_to_num(control.steer),
            'throttle': np.nan_to_num(control.throttle),
            'brake': np.nan_to_num(control.brake),
            'hand_brake': control.hand_brake,
            'reverse': control.reverse,
            'waypoint': [ego_waypoint.transform.location.x, ego_waypoint.transform.location.y, ego_waypoint.transform.location.z]
        }

class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider, writer=None, global_plan=None):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)
        self._writer = writer
        self._global_plan=global_plan

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag, self._writer)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag, self._writer)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag, self._writer)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag, self._global_plan, self._writer)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag, self._writer)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag, self._writer)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag, writer):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._data_provider.update_sensor(image, tag, array, image.frame_number, writer)

    def _parse_lidar_cb(self, lidar_data, tag, writer):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(lidar_data, tag, points, lidar_data.frame, writer)

    def _parse_radar_cb(self, radar_data, tag, writer):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(radar_data, tag, points, radar_data.frame, writer)

    def _parse_gnss_cb(self, gnss_data, tag, global_plan, writer):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        if self._data_provider._global_plan is None:
            self._data_provider.setup_global_plan(global_plan)
        self._data_provider.update_sensor(gnss_data, tag, array, gnss_data.frame, writer)

    def _parse_imu_cb(self, imu_data, tag, writer):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(imu_data, tag, array, imu_data.frame, writer)

    def _parse_pseudosensor(self, package, tag, writer):
        self._data_provider.update_sensor(None, tag, package.data, package.frame, writer)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10
        self._written = {}
        self._lock = Lock()
        self._direction_planner=None
        self._global_plan=None

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None

    def setup_global_plan(self, global_plan):
        self._global_plan = global_plan

    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor
        self._data_buffers[tag] = None

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, raw, tag, data, timestamp, writer):
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))
        self._data_buffers[tag] = data
        self._synchronize_write(tag, raw, data, writer)

    def get_data(self):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict

    def all_sensors_ready(self):
        for key in self._sensors_objects.keys():
            if self._data_buffers[key] is None:
                return False
        # Sensors are ready initialize written as 0
        if not self._written:
            for key in self._sensors_objects.keys():
                self._written[key] = 0
        return True

    def wait_sensors_written(self, writer):
        unsynchronized = True

        while unsynchronized:
            unsynchronized = False
            for tag in self._written.keys():
                if self._written[tag] <=  writer._latest_id:
                    unsynchronized= True

            time.sleep(0.01)

    def _synchronize_write(self, tag, raw, data, writer):
        """
        Synchronize to check if all sensors have been written.
        """

        if writer is not None:
            if not self.all_sensors_ready():
                return
            self._lock.acquire()
            if self._written[tag] > writer._latest_id:
                self._lock.release()
                return

            if tag.startswith('rgb'):
                writer.write_image(raw, tag)
            elif tag in ['GPS']:
                if self._direction_planner is None:
                    self._direction_planner = Waypointer(self._global_plan, data)
                _, _, cmd = self._direction_planner.tick(data, self._data_buffers['IMU'])
                wp_data = {'direction': float(cmd.value)}
                writer.write_gnss(wp_data, 'can_bus')
            elif tag in ['can_bus', 'SPEED']:
                writer.write_pseudo(data, 'can_bus')
            else:
                pass

            self._written[tag] += 1
            self._lock.release()

        else:
            return

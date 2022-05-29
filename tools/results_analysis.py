
import os
import glob
import re
import json
import math
import carla
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import carla
import traceback
import scipy.misc
from scipy.interpolate import interp1d
import shutil
import time
import h5py

from PIL import Image, ImageDraw, ImageFont
from tools.utils import draw_map, draw_point_data
from benchmark.utils.server_manager import ServerManagerDocker, find_free_port


COLOR_BUTTER_0 = (252/ 255.0, 233/ 255.0, 79/ 255.0)
COLOR_BUTTER_1 = (237/ 255.0, 212/ 255.0, 0/ 255.0)
COLOR_BUTTER_2 = (196/ 255.0, 160/ 255.0, 0/ 255.0)

COLOR_ORANGE_0 = (252/ 255.0, 175/ 255.0, 62/ 255.0)
COLOR_ORANGE_1 = (245/ 255.0, 121/ 255.0, 0/ 255.0)
COLOR_ORANGE_2 = (209/ 255.0, 92/ 255.0, 0/ 255.0)

COLOR_CHOCOLATE_0 = (233/ 255.0, 185/ 255.0, 110/ 255.0)
COLOR_CHOCOLATE_1 = (193/ 255.0, 125/ 255.0, 17/ 255.0)
COLOR_CHOCOLATE_2 = (143/ 255.0, 89/ 255.0, 2/ 255.0)

COLOR_CHAMELEON_0 = (138/ 255.0, 226/ 255.0, 52/ 255.0)
COLOR_CHAMELEON_1 = (115/ 255.0, 210/ 255.0, 22/ 255.0)
COLOR_CHAMELEON_2 = (78/ 255.0, 154/ 255.0, 6/ 255.0)

COLOR_GREEN_0 = (0.0/ 255.0, 255.0/ 255.0, 0.0/ 255.0)

COLOR_SKY_BLUE_0 = (114/ 255.0, 159/ 255.0, 207/ 255.0)
COLOR_SKY_BLUE_1 = (52/ 255.0, 101/ 255.0, 164/ 255.0)
COLOR_SKY_BLUE_2 = (32/ 255.0, 74/ 255.0, 135/ 255.0)

COLOR_PLUM_0 = (173/ 255.0, 127/ 255.0, 168/ 255.0)
COLOR_PLUM_1 = (117/ 255.0, 80/ 255.0, 123/ 255.0)
COLOR_PLUM_2 = (92/ 255.0, 53/ 255.0, 102/ 255.0)

COLOR_SCARLET_RED_0 = (239/ 255.0, 41/ 255.0, 41/ 255.0)
COLOR_SCARLET_RED_1 = (204/ 255.0, 0/ 255.0, 0/ 255.0)
COLOR_SCARLET_RED_2 = (164/ 255.0, 0/ 255.0, 0/ 255.0)

COLOR_ALUMINIUM_0 = (238/ 255.0, 238/ 255.0, 236/ 255.0)
COLOR_ALUMINIUM_1 = (211/ 255.0, 215/ 255.0, 207/ 255.0)
COLOR_ALUMINIUM_2 = (186/ 255.0, 189/ 255.0, 182/ 255.0)
COLOR_ALUMINIUM_3 = (136/ 255.0, 138/ 255.0, 133/ 255.0)
COLOR_ALUMINIUM_4 = (85/ 255.0, 87/ 255.0, 83/ 255.0)
COLOR_ALUMINIUM_4_5 = (66/ 255.0, 62/ 255.0, 64/ 255.0)
COLOR_ALUMINIUM_5 = (46/ 255.0, 52/ 255.0, 54/ 255.0)

COLOR_WHITE = (255/ 255.0, 255/ 255.0, 255/ 255.0)
COLOR_BLACK = (0/ 255.0, 0/ 255.0, 0/ 255.0)
COLOR_LIGHT_GRAY = (196/ 255.0, 196/ 255.0, 196/ 255.0)
COLOR_PINK = (255/255.0,192/255.0,203/255.0)



def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def compute_relative_angle(vehicle_loc,vehicle_rot, waypoint_loc):
    v_begin = carla.Location(x=vehicle_loc[0], y=vehicle_loc[1], z=vehicle_loc[2])

    v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_rot[1])),
                                     y=math.sin(math.radians(vehicle_rot[1])))

    v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    w_vec = np.array([waypoint_loc[0] -
                      v_begin.x, waypoint_loc[1] -
                      v_begin.y, 0.0])

    relative_angle = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
    _cross = np.cross(v_vec, w_vec)
    if _cross[2] < 0:
        relative_angle *= -1.0

    if np.isnan(relative_angle):
        relative_angle = 0.0

    return relative_angle

def compute_values(data_paths, save_path, scenario_name, value_name, make_plot=True):
    color_id = 0
    colors_list=['b','g','m', 'c', 'y']
    if make_plot:
        plt.figure()
    for data_path in data_paths:
        print("  ")
        print("----------- ", save_path.split('/')[-1], "----------- ")
        json_path_list = glob.glob(os.path.join(data_path, 'can_bus*.json'))
        sort_nicely(json_path_list)

        if value_name == 'speed':
            ego_speeds = []
            #obstacle_speed=[]
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    ego_speeds.append(np.clip(round(data['speed'], 1), 0.0, None))
                    #obstacle_speed.append(np.clip(round(data[scenario_name]['speed_obstacle1'], 1), 0.0, None))

            if make_plot:
                plt.plot(range(len(json_path_list)), ego_speeds, '-', color=colors_list[color_id])
                #plt.plot(range(len(json_path_list)), obstacle_speed, '-', color=colors_list[color_id+1])
                plt.xlim([0, len(json_path_list)])
                #plt.xlim([200, 600])
                plt.title('Driving Velocity')
                #plt.ylim([3, 10])

            long_comfort=[]
            for i in range(len(ego_speeds)-1):
                long_comfort_value = abs(ego_speeds[i+1]-ego_speeds[i])/0.1
                long_comfort.append(long_comfort_value)

            print('Comfort Value: long', np.mean(long_comfort), np.std(long_comfort))
            #print(np.mean(ego_speeds))

        elif value_name == 'relative_angle':
            ego_info=[]
            wp = []
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    ego_info.append((data['ego_location'], data['ego_rotation']))
                    wp.append(data['ego_wp_location'])
            relative_angle=[]
            for i in range(len(ego_info)-1):
                ego_loc, ego_rot = ego_info[i]
                next_wp = wp[i+1]
                relative_angle.append(compute_relative_angle(ego_loc, ego_rot, next_wp))

            if make_plot:
                plt.xlim([0, len(json_path_list)-1])
                plt.plot(range(len(json_path_list)-1), relative_angle, '-', color=colors_list[color_id])
                plt.title('Relative Angle')

            lat_comfort = []
            for i in range(len(relative_angle) - 1):
                lat_comfort_value = abs(relative_angle[i + 1] - relative_angle[i]) / 0.1
                lat_comfort.append(lat_comfort_value)

            print('Comfort Value: lat', np.mean(lat_comfort), np.std(lat_comfort))

        elif value_name == 'acc':
            acc=[]
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    acc.append(data['throttle'] if not data['throttle'] == 0.0 else -data['brake'])
            if make_plot:
                plt.xlim([0, len(json_path_list)])
                plt.plot(range(len(json_path_list)), acc, '-', color=colors_list[color_id])
                plt.title('Acceleration Prediction')

        elif value_name == 'steering':
            steer=[]
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    steer.append(data['steer'])
            if make_plot:
                plt.xlim([0, len(json_path_list)])
                plt.plot(range(len(json_path_list)), steer, '-', color=colors_list[color_id])
                plt.title('Steering Value Prediction')

        elif value_name == 'dist':
            distances=[]
            leading_actor_speeds = []
            x=[]
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    leading_actor_speed = np.clip(round(data[scenario_name]['speed_obstacle1'], 1), 0.0, None)
                    # count when the leading car starts
                    if leading_actor_speed >0.0:
                        try:
                            leading_actor_speeds.append(leading_actor_speed)
                            leading_actor_location = data[scenario_name]['obstacle1_location']
                            ego_location = data['ego_location']
                            distance_vector = carla.Location(x=ego_location[0], y=ego_location[1],
                                                             z=0.0) - carla.Location(x=leading_actor_location[0],
                                                                                     y=leading_actor_location[1], z=0.0)
                            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))
                            distances.append(distance)
                            x.append(0 if not json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')
                                     else int(json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')))
                        except:
                            pass

            leading_stop_id = leading_actor_speeds[100:].index(0.0)
            sample_num_front = 20
            sample_num_back=20

            leading_stop_frame = x[leading_stop_id]


            if make_plot:
                plt.axvline(x = leading_stop_frame, color=colors_list[color_id])

                plt.title('Distance to Leading Vehicle')
                plt.plot(x[leading_stop_id-sample_num_front:leading_stop_id+sample_num_back],
                         distances[leading_stop_id-sample_num_front:leading_stop_id+sample_num_back], color=colors_list[color_id])
                #plt.xlim([x[0], x[-1]])
                #plt.ylim([7, 20])

        elif value_name == 'relative_speed':
            relative_speed = []
            leading_actor_speeds=[]
            abs_relative_speed=[]
            x=[]
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    leading_actor_speed = np.clip(data[scenario_name]['speed_obstacle1'], 0.0, None)
                    ego_speed = np.clip(data['speed'], 0.0, None)
                    # count when the leading car starts
                    if leading_actor_speed > 0.0:
                        try:
                            leading_actor_speeds.append(leading_actor_speed)
                            relative_speed.append((ego_speed - leading_actor_speed))
                            x.append(0 if not json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')
                                     else int(json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')))
                        except:
                            pass

            leading_stop_id = leading_actor_speeds[100:].index(0.0)
            leading_stop_frame = x[leading_stop_id]

            if make_plot:
                plt.axvline(x=leading_stop_frame, color=colors_list[color_id])
                plt.plot(x,relative_speed, '-', color=colors_list[color_id])
                plt.plot(range(len(json_path_list)), np.zeros(len(json_path_list)).tolist(), color='k')
                plt.xlim([x[0], x[-1]])
                plt.title('Relative Speed')
                #plt.ylim([-3, 4])

        elif value_name == 'ttc':
            TTC = []
            x = []
            leading_x = []
            leading_actor_speeds = []
            for json_file in json_path_list:
                with open(json_file) as json_:
                    data = json.load(json_)
                    leading_actor_speed = np.clip(round(data[scenario_name]['speed_obstacle1'], 1), 0.0, None)
                    # count when the leading car starts
                    if leading_actor_speed > 0.0:
                        try:
                            leading_actor_speeds.append(leading_actor_speed)
                            leading_x.append(0 if not json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')
                                             else int(json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')))
                            leading_actor_location = data[scenario_name]['obstacle1_location']
                            ego_location = data['ego_location']
                            distance_vector = carla.Location(x=ego_location[0], y=ego_location[1],
                                                             z=0.0) - carla.Location(x=leading_actor_location[0],
                                                                                     y=leading_actor_location[1], z=0.0)
                            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))
                            if (np.clip(round(data['speed'], 1), 0.0, None) - np.clip(round(leading_actor_speed, 1), 0.0, None)) > 0.0:
                                ttc = distance / (np.clip(round(data['speed'], 1), 0.0, None) - np.clip(round(leading_actor_speed, 1), 0.0, None))
                                TTC.append(ttc)
                                x.append(0 if not json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')
                                         else int(json_file.split('/')[-1].split('.')[-2][-6:].lstrip('0')))
                        except:
                            pass
            print(np.mean(leading_actor_speeds))
            leading_stop_id = leading_actor_speeds[100:].index(0.0)
            leading_stop_frame = leading_x[leading_stop_id]
            plt.axvline(x=leading_stop_frame, color=colors_list[color_id])
            sample_num_front = 20
            sample_num_back = 20

            valid_point_frame_id = [x.index[p] for p in x if p >= leading_stop_frame-sample_num_front and p<= leading_stop_frame+sample_num_back]
            valid_ttc = [TTC[frame_id] for frame_id in valid_point_frame_id]

            x = np.array(x)
            y = np.array(TTC)
            x_new = np.linspace(min(x), max(x), max(x)-min(x))
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            x_new = [int(x) for x in x_new]
            leading_stop_frame_in_ttc = x_new.index(leading_stop_frame)

            if make_plot:
                #plt.plot(x, y, color=colors_list[color_id])
                #plt.plot(x, np.zeros(len(x)).tolist(), color='k')
                plt.title('Time to Collision')
                plt.plot(x_new[leading_stop_frame_in_ttc-sample_num_front: leading_stop_frame_in_ttc+sample_num_back],
                         y_smooth[leading_stop_frame_in_ttc-sample_num_front: leading_stop_frame_in_ttc+sample_num_back], color=colors_list[color_id])
                plt.plot(x_new[leading_stop_frame_in_ttc-sample_num_front: leading_stop_frame_in_ttc+sample_num_back],
                         np.zeros(len(x_new[leading_stop_frame_in_ttc-sample_num_front: leading_stop_frame_in_ttc+sample_num_back])).tolist(), color='k')
                #plt.ylim([0, 100])

        color_id += 1

    if make_plot:
        plt.xlabel('frame id')
        plt.ylabel('')

        plt.savefig(os.path.join(save_path + '_'+value_name+'.png'))
        plt.close()



def draw_trajectory_debug(data_paths, save_path, world):
    trajectories_fig = plt.figure(0)
    draw_map(world)
    #plt.figure()
    #colors_list = ['b', 'g', 'm', 'c', 'y']
    colors_list = [COLOR_PINK, COLOR_GREEN_0, COLOR_ORANGE_0, COLOR_BUTTER_0]
    color_id = 0
    min_pixels_x = 100000000
    max_pixels_x = -100000000
    min_pixels_y = 100000000
    max_pixels_y = -100000000
    for data_path in data_paths:
        print(data_path.split('/')[-2])
        x=[]
        y=[]
        pixels_x=[]
        pixels_y = []
        json_path_list = glob.glob(os.path.join(data_path, 'can_bus*.json'))
        sort_nicely(json_path_list)
        for json_file in json_path_list:
            with open(json_file) as json_:
                data = json.load(json_)
                ego_location = data['ego_location']
                x.append(ego_location[0])
                y.append(ego_location[1])

                datapoint = [ego_location[0], ego_location[1], ego_location[2]]
                pixel = draw_point_data(datapoint, trajectories_fig, color=colors_list[color_id], size=5)
                pixels_x.append(pixel[0])
                pixels_y.append(pixel[1])

        #plt.scatter(x, y, color=colors_list[color_id], s=5)
        color_id += 1
        if min(pixels_x) < min_pixels_x:
            min_pixels_xx = min(pixels_x)
            min_pixels_xy = pixels_y[pixels_x.index(min(pixels_x))]
        if max(pixels_x) > max_pixels_x:
            max_pixels_xx = max(pixels_x)
            max_pixels_xy = pixels_y[pixels_x.index(max(pixels_x))]

        if min(pixels_y) < min_pixels_y:
            min_pixels_yy = min(pixels_y)
            min_pixels_yx = pixels_x[pixels_y.index(min(pixels_y))]
        if max(pixels_y) > max_pixels_y:
            max_pixels_yy = max(pixels_y)
            max_pixels_yx = pixels_x[pixels_y.index(max(pixels_y))]

    plt.xlim(-250, 2500)
    plt.ylim(1000, 4000)
    trajectories_fig.savefig(os.path.join(save_path + '_' + 'map.png'))

    plt.xlim([min(min_pixels_xx, min_pixels_yx) -200, max(max_pixels_xx, max_pixels_yx) + 200])
    plt.ylim([min(min_pixels_xy, min_pixels_yy) -200, max(max_pixels_yy, max_pixels_xy) + 200])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('location')
    trajectories_fig.savefig(os.path.join(save_path + '_' + 'trajectory.png'))
    plt.close()

def inverse_pose(pose):
    inv_pose = np.eye(3)
    Rcw = pose[0:2,0:2]
    tcw = pose[0:2,2]
    Rwc = Rcw.transpose()
    Ow  = np.dot(-Rwc, tcw)
    inv_pose[0:2,0:2] = Rwc
    inv_pose[0:2,2] = Ow
    return inv_pose

def compute_comfort_values(data_path, no_leading=False):
    routes = glob.glob(os.path.join(data_path, '*'))
    sort_nicely(routes)
    long_comfort_values=[]
    lat_comfort_values = []
    for route_path in routes:
        json_path_list = glob.glob(os.path.join(route_path, '0','can_bus*.json'))
        sort_nicely(json_path_list)
        ego_speeds = []
        ego_info = []
        wp = []
        obstacle_info = []
        for json_file in json_path_list:
            with open(json_file) as json_:
                data = json.load(json_)
                if not no_leading:
                    try:
                        obstacle_info.append((data['FollowingLeadingVehicleInLowSpeed']['obstacle1_location'],
                                              data['FollowingLeadingVehicleInLowSpeed']['obstacle1_rotation']))
                        ego_speeds.append(np.clip(round(data['speed'], 1), 0.0, None))
                        ego_info.append((data['ego_location'], data['ego_rotation']))

                        wp.append(data['ego_wp_location'])
                    except:
                        pass
                else:
                    ego_speeds.append(np.clip(round(data['speed'], 1), 0.0, None))
                    ego_info.append((data['ego_location'], data['ego_rotation']))
                    wp.append(data['ego_wp_location'])

        long_comfort = []
        for i in range(len(ego_speeds) - 1):
            if ego_speeds[i] > 0.00:
                if not no_leading:
                    obstacle_to_world = np.asarray([[obstacle_info[i][0][0]],
                                                    [obstacle_info[i][0][1]],
                                                    [1.0]])
                    yaw_radiant = np.deg2rad(ego_info[i][1][1])
                    pose_ego_to_world = np.asarray([[np.cos(yaw_radiant), -np.sin(yaw_radiant), ego_info[i][0][0]],
                                                    [np.sin(yaw_radiant),  np.cos(yaw_radiant), ego_info[i][0][1]],
                                                    [0.0, 0.0, 1.0]])

                    pose_world_to_ego = inverse_pose(pose_ego_to_world)
                    obstacle_to_ego = np.dot(pose_world_to_ego, obstacle_to_world)

                    if obstacle_to_ego[0] >0:
                        long_comfort_value = abs(ego_speeds[i + 1] - ego_speeds[i]) / 0.1
                        long_comfort.append(long_comfort_value)
                    else:
                        print(route_path.split('/')[-1],i)
                else:
                    long_comfort_value = abs(ego_speeds[i + 1] - ego_speeds[i]) / 0.1
                    long_comfort.append(long_comfort_value)

        long_comfort_values += long_comfort

        relative_angle = []
        for i in range(len(ego_info) - 1):
            ego_loc, ego_rot = ego_info[i]
            next_wp = wp[i + 1]
<<<<<<< HEAD
            ra = compute_relative_angle(ego_loc, ego_rot, next_wp)
=======
            cur_wp = wp[i]
            ra = compute_relative_angle(ego_loc, ego_rot, cur_wp, next_wp)
>>>>>>> 02490ce77f22e32b84c950447de2710af5c7fd20
            relative_angle.append(ra)

        lat_comfort = []
        for i in range(len(relative_angle) - 1):
            lat_comfort_value = abs(relative_angle[i + 1] - relative_angle[i]) / 0.1
            lat_comfort.append(lat_comfort_value)

        lat_comfort_values += lat_comfort

    print(' longtitude:', np.mean(long_comfort_values), np.std(long_comfort_values))
    print(' ')
    print(' lateral:', np.mean(lat_comfort_values), np.std(lat_comfort_values))
    print(' ')



make_plots = False
make_trajectries = False
analysis_comfort_values =True

if analysis_comfort_values:
    root_dir = os.path.join(os.environ['SENSOR_SAVE_PATH'],'Scenario5_newweathertown_Town02')
    model_name = '20220405_TempoarlTFM_EnDe_5Frames_Roach19Hours_T1_seed1_1_nomask_finetune_cmd_600000_10FPS'
<<<<<<< HEAD
    car_type= ['carlacola']
=======
    car_type= ['microlino']
>>>>>>> 02490ce77f22e32b84c950447de2710af5c7fd20
    leading_speed= ['3mps']

    #car_type = ['noleadingcar']
    #leading_speed = ['none']

    for car in car_type:
        for sp in leading_speed:
            data_path = os.path.join(root_dir, model_name, car, sp)
            print('-----------------------------------------------------')
            print('Model:', model_name)
            print(car, sp)
            compute_comfort_values(data_path, no_leading= True if car == 'noleadingcar' else False)


if make_plots:
    value_names= ['speed','relative_angle']
    route_id = range(28)
    root_dir = os.path.join(os.environ['SENSOR_SAVE_PATH'],'Scenario5_newweathertown_Town02')
    car_type= 'carlacola'
    leading_speed='2mps'

    for i in route_id:
        route_name = 'RouteScenario_' + str(i) + '_Scenario5'
        for value_name in value_names:
            data_paths = [
                os.path.join(root_dir,'20220405_TempoarlTFM_EnDe_5Frames_Roach19Hours_T1_seed1_1_nomask_finetune_cmd_600000_10FPS', car_type, leading_speed, route_name,'0'),
                #os.path.join(root_dir,'20220405_TempoarlTFM_En_5Frames_LastAction_NoToken_Roach19Hours_T1_seed1_1_nomask_finetune_cmd_650000_10FPS', car_type, leading_speed, route_name,'0'),
                #root_dir+'20220405_Roach_rl_birdview_11833344_10FPS/' + route_name + '/0'
                ]
            save_path = os.path.join(root_dir, car_type+'_'+leading_speed+'_'+route_name)
            compute_values(data_paths, save_path,
                    'SignalJunctionLeadingVehicleCrossingRedTrafficLight',
                    value_name, make_plot=False)

if make_trajectries:
    #params = {'docker_name': 'carlasim/carla:0.9.13', 'gpu': 0, 'quality_level': 'Epic'}
    #ServerDocker = ServerManagerDocker(params)
    #ServerDocker.reset(port=2000)
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    # Here we set up the world
    carla_world = client.load_world('Town02')
    print('loaded world!')

    root_dir = '/datatmp/Datasets/yixiao/CARLA/SelfDefined_Scenarios/Scenario2_newSingleweathertown_Town02_test/'

    route_id = range(32)

    for i in route_id:
        route_name = 'RouteScenario_' + str(i) + '_Scenario2'
        data_paths = [
            root_dir+'20220405_SingleFrame_Roach19Hours_T1_seed1_1_500000_10FPS/' + route_name + '/0',
            root_dir+'20220405_FramesStacking_5Frames_Roach19Hours_T1_seed1_1_500000_10FPS/' + route_name + '/0',
            root_dir+'20220405_TempoarlTFM_En_5Frames_Roach19Hours_T1_seed1_1_mask_500000_10FPS/' + route_name + '/0',
            #root_dir+'20220405_TempoarlTFM_En_10Frames_Roach19Hours_T1_seed1_1_mask_200000_10FPS/' + route_name + '/0',
            #root_dir+'20220405_TempoarlTFM_EnDe_5Frames_Roach19Hours_T1_seed1_1_nomask_500000_10FPS/' + route_name + '/0',
            #root_dir + '20220405_TempoarlTFM_EnDe_10Frames_Roach19Hours_T1_seed1_1_nomask_200000_10FPS/' + route_name + '/0',
            #root_dir+'20220405_TempoarlTFM_En_5Frames_LastAction_NoToken_Roach19Hours_T1_seed1_1_nomask_500000_10FPS/' + route_name + '/0',
            #root_dir + '20220405_TempoarlTFM_En_10Frames_LastAction_NoToken_Roach19Hours_T1_seed1_1_nomask_500000_10FPS/' + route_name + '/0',
            root_dir+'20220405_Roach_rl_birdview_11833344_10FPS/' + route_name + '/0',
        ]
        save_path = root_dir + '5F_AllActions_'+route_name
        draw_trajectory_debug(data_paths, save_path, world=carla_world)




def draw_trajectory(data_root, world, town, route_trajectory):
    trajectories_fig = plt.figure(0)
    draw_map(world)
    colors_list = [COLOR_PINK, COLOR_GREEN_0]
    json_path_list = glob.glob(os.path.join(data_root, 'can_bus*.json'))
    sort_nicely(json_path_list)
    for json_file in json_path_list:
        with open(json_file) as json_:
            data = json.load(json_)
            start_point = [route_trajectory[0].x, route_trajectory[0].y, route_trajectory[0].z]
            _ = draw_point_data(start_point, trajectories_fig, color=(0/ 255.0, 0/ 255.0, 255/ 255.0), size=30)
            end_point = [route_trajectory[-1].x, route_trajectory[-1].y, route_trajectory[-1].z]
            _ = draw_point_data(end_point, trajectories_fig, color=(252/ 255.0, 175/ 255.0, 62/ 255.0), size=20)
            ego_location = data['ego_location']
            datapoint = [ego_location[0], ego_location[1], ego_location[2]]
            _ = draw_point_data(datapoint, trajectories_fig, color=COLOR_GREEN_0, size=20)
            # TODO: HARDCODING
            try:
                actor_location = data['FollowingLeadingVehicleInLowSpeed']['obstacle1_location']
                datapoint = [actor_location[0], actor_location[1], actor_location[2]]
                _ = draw_point_data(datapoint, trajectories_fig, color=COLOR_PINK, size=20)
            except:
                pass

    if town == 'Town01':
        plt.xlim(-500, 5000)
        plt.ylim(-500, 4500)
    elif town == 'Town02':
        plt.xlim(-250, 2500)
        plt.ylim(1000, 4000)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(data_root.split('/')[-2])
    trajectories_fig.savefig(os.path.join('/'.join(data_root.split('/')[:-1]) + '_' + 'trajectory.png'))
    plt.close()
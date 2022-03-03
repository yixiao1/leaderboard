#!/bin/bash

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--debug=0 \
--scenarios=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json  \
--routes=${LEADERBOARD_ROOT}/data/routes_devtest.xml \
--repetitions=1 \
--track=SENSORS \
--checkpoint=${LEADERBOARD_ROOT}/results/results_20210910_TFM_SpeedLoss_im_res34_CARLA_T1346_1CAM_seed1_1_1.json \
--agent=${LEADERBOARD_ROOT}/leaderboard/autoagents/TFM_Encoder_agent.py \
--agent-config=${TRAINING_ROOT}/_results/Encoders_20210910/TFM_SpeedLoss_im_res34_CARLA_T1346_1CAM_seed1_1/config.json \
--docker=carla_0910_t10:0.9.10 \
--gpus=6,7 \
--save-sensors=/datatmp/Datasets/yixiao/CARLA/leaderboard/routes_devtest/20210909_TFM_PS_im_res34_CARLA_T1346_1CAM_seed1_1_800000

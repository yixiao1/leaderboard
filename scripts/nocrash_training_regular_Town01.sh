#!/bin/bash

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--debug=0 \
--scenarios=${LEADERBOARD_ROOT}/data/nocrash/nocrash_training_regular_Town01.json  \
--routes=${LEADERBOARD_ROOT}/data/nocrash/Town01_navigation.xml \
--repetitions=1 \
--resume=true \
--track=SENSORS \
--checkpoint=${LEADERBOARD_ROOT}/results/nocrash \
--agent=${LEADERBOARD_ROOT}/leaderboard/autoagents/FramesStacking_SpeedInput_agent.py \
--agent-config=${TRAINING_RESULTS_ROOT}/_results/20211201/SingleFrame_SpeedLossInput_im_res34_Roach_T1_1CAM_seed1_acc_1/config.json \
--docker=carla_0910_t10:0.9.10 \
--gpus=8,9 \
--fps=10 \

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:20:03 2020

@author: aznarr1
"""

import gym
import numpy as np

import highway_env

from highway_env.envs.multi_agent_intersection_env import MAIntersectionEnv
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv

"""
        2
        |
        |
1 ------ ------3
        |
        |
        0
"""

# n_learning_agents = 4 # at the moment max learning agents is 4. 

# config = {
#     "observation": {
#             "type": "Kinematics",
#             # "vehicles_count": n_learning_agents,
#             "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#             "features_range": {
#                 "x": [-100, 100],
#                 "y": [-100, 100],
#                 "vx": [-20, 20],
#                 "vy": [-20, 20],
#             },
#             "absolute": True,
#             "flatten": False,
#             "observe_intentions": False,
#         },
#     "offscreen_rendering": False ,
#     # "duration": 100,
#     # "controlled_vehicles_count": n_learning_agents,
#     # "normalize_reward": False,
#     ## if you want to set the start and end of learning agents, use start_positions and end_positions to set them according the above figure. 
#     ## for example --> unprotected left turn scenario: n_learning_agents = 2,  "start_positions": [0, 2] , "end_positions": [1, 0]  
#     # "auto_select_starts_ends": True,
#     ## if you don't wnat any other random vehicle, set the followings to zero
#     "initial_vehicle_count": 0, 
#     "spawn_probability": 0
# }
# env = MAIntersectionEnv(config=config)



config = {
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "lateral": False,
            "longitudinal": True
        }
    },
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": True,
            "order": "shuffled"
        },
    },
    "controlled_vehicles": 2,
    "initial_vehicle_count": 0, 
    "spawn_probability": 0
}

env = MultiAgentIntersectionEnv(config=config)

done = False
while True:
    action = (1, 1) # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
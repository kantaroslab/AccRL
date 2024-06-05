# Accelerated Reinforcement Learning

## How to use the code 

#### Configuration
All the task related configuration is located in `./utils/params.yaml`
* LTL Task: `ltl_task` and `obstacle_list`
* Grid Size: `grid_size`
* Number of Episodes: `episodes`
* Max Steps per Episode: `max_steps`
* Parameter A of Biased 2 & 3: `our2_A` and `our3_A`

#### System Requirement

This repo needs to be run in Linux system, we tested on Ubuntu 20.04 using Python 3.8

## Methods

#### Our Method

`python3 main.py --type 1`, where `1` can be replaced with 1, 2, 3, 4, and 5, representing different parameter choice of our method shown in paper.

#### Epsilon-Greedy Method

`python3 random_explore.py`

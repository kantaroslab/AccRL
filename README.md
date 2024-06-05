# Accelerated Reinforcement Learning

## How to use the code 

#### Configuration
All the task related configuration is located in `./utils/params.yaml`
* LTL Task: `ltl_task` and `obstacle_list`
* Grid Size: `grid_size`
* Number of Episodes: `episodes`
* Max Steps per Episode: `max_steps`

#### System Requirement

This repo needs to be run in Linux system, we tested on Ubuntu 20.04 using Python 3.8

## Methods

#### Our Method

`python3 main.py --type 1`, where `1` can be replaced with 1~6, representing different decay rate of our method shown in paper.

* Notice LTL formula would not change when pick different type, only decay rate would change
* `--type`
    * Small MDP
        * 1: Biased-1
        * 2: Biased-2
        * 3: Biased-3
    * Big MDP
        * 4: Biased-1
        * 5: Biased-2
        * 6: Biased-3

#### Epsilon-Greedy Method

`python3 random_explore.py`

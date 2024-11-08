# Sample-Efficient Reinforcement Learning with Temporal Logic Objectives: Leveraging the Task Specification to Guide Exploration

## How to use the code 

#### Input
The inputs to the reinforcement learning algorithm are located in `./utils/params.yaml` that include:
* LTL Task: `ltl_task` and `obstacle_list`
* Grid size: `grid_size` (e.g., if grid_size=50, then the MDP models a 50x50 grid world resulting in an MDP with 2,500 states)
* Number of episodes: `episodes`
* Maximum number of steps per episode: `max_steps`

#### System Requirement

This repo was implemented using using Python 3.8 and tested on Ubuntu 20.04.

#### Our Method
To run our method, type:

`python3 main.py --type X`, where `X` can take values from {1,2,3,4,5,6}. The value of X represents different decay rates of the parameters \epsilon, \delta_b, and \delta_e of our method. 

* Values of X
  * X = 1: Corresponds to Biased-1 used in [A] for the 10x10 and 20x20 MDPs
  * X = 2: Corresponds to Biased-2 used in [A] for the 10x10 and 20x20 MDPs
  * X = 3: Corresponds to Biased-3 used in [A] for the 10x10 and 20x20 MDPs
  * X = 4: Corresponds to Biased-1 used in [A] for the 50x50 MDP
  * X = 5: Corresponds to Biased-2 used in [A] for the 50x50 MDP
  * X = 6: Corresponds to Biased-3 used in [A] for the 50x50 MDP

#### Epsilon-Greedy Method
To run our method with the biased part disabled (i.e., \delta_b=0), use the following command
`python3 random_explore.py`

#### Output
The code will return a deterministic policy (greedy with respect to the Q value function) along with its satisfaction probability. The latter is computed using the unknown-to-the-system MDP transition probabilities.

#### Reference
[A] Y. Kantaros, and J. Wang. "Sample-Efficient Reinforcement Learning with Temporal Logic Objectives: Leveraging the Task Specification to Guide Exploration." IEEE Transactions on Automatic Control (2024).




import torch
import time
import csv
import math
import argparse
import shutil
import numpy as np
from utils.helper import * 
from utils.translate import *


dijkstra_map = {}
parser = argparse.ArgumentParser(description='Parameter setup for Biased Method')
parser.add_argument('--type', type=int, default=1)
parser.add_argument('--para', type=int, default=1)
args = parser.parse_args()

def bias_action(mix_state, grid_size, rabin, gridworld, TP):
    mdp_state = mix_state[0]
    x, y = index_to_coordinates(mdp_state, grid_size)
    ur = set()
    if y + 1 >= grid_size or x-1 < 0:
        ur = (x, y)
    else:
        ur = (x-1, y+1)
    dl = set()
    if x + 1 >= grid_size or y-1 < 0:
        dl = (x, y)
    else:
        dl = (x+1, y-1)
    ul = set()
    if y-1 < 0 or x-1 < 0:
        ul = (x, y)
    else:
        ul = (x-1, y-1)
    dr = set()
    if x+1 >= grid_size or y+1>= grid_size:
        dr = (x, y)
    else:
        dr = (x+1, y+1)

    next_states = {
        0: (x, max(0, y - 1)), # left
        1: (x, min(grid_size - 1, y + 1)), # right 
        2: (max(0, x - 1), y), # up
        3: (min(grid_size - 1, x + 1), y), # down
        4: (ur[0], ur[1]), # up right
        5: (dl[0], dl[1]), # down left
        6: (ul[0], ul[1]), # up left,
        7: (dr[0], dr[1]), # down right
        8: (x, y), # unchanged
    }

    # find one-hop reachable new qt
    goal_mdp, _ = rabin.select_next_goal(mix_state)
    goal_mdp = index_to_coordinates(goal_mdp, grid_size)
    new_grid = gridworld

    # filter out the adjoint MDP states that show shortest Dijkstra distance
    min_distance = float('inf')
    min_elements = []
    min_l2_dis = float('inf')
    for next_mdp in list(set(next_states.values())):
        if coordinates_to_index(next_mdp[0], next_mdp[1], grid_size) in rabin.coord_dict['obstacles']:
            continue 
        l2_dis = np.sqrt((goal_mdp[0] - next_mdp[0])**2 + (goal_mdp[1]-next_mdp[1])**2)

        if (next_mdp, goal_mdp) not in dijkstra_map:
            length = dijkstra(new_grid, next_mdp, goal_mdp) - 1
            dijkstra_map[(next_mdp, goal_mdp)] = length
        else:
            length = dijkstra_map[(next_mdp, goal_mdp)]
        

        if length < min_distance or (length == min_distance and l2_dis < min_l2_dis):
            min_distance = length
            min_l2_dis = l2_dis
            min_elements = [coordinates_to_index(next_mdp[0], next_mdp[1], grid_size)]
    max_val = -1
    res = None
    for next_mdp in min_elements:
        cur = np.max(TP[mdp_state, next_mdp])
        if cur > max_val:
            max_val = cur
            best_next = next_mdp
    bx, by = index_to_coordinates(best_next, grid_size)
    for key, val in next_states.items():
        if val == (bx, by):
            res = key
    return int(res)    


def main():
    clean_files(os.getcwd())
    # Load configuration & Setup output folder
    if args.para == 1:
        config = load_config('./utils/params.yaml')
    elif args.para == 2:
        config = load_config('./utils/params_new.yaml')
    discount_factor = config['discount_factor']
    logging, writer, output_folder = log_init(config, type='ours', our_param_type=args.type)
    reward_file_name = os.path.join(output_folder, "discount_episode_reward.csv")
    
    # Define grid and action space 
    grid_size = config['grid_size']
    num_states = config['grid_size'] * config['grid_size']
    num_actions = config['num_actions'] # left, right, up, down, stay idle

    # Initialize DRA 
    rabin = rabin_setup(config)
    num_nodes = rabin.num_of_nodes
    move2result(['command*'], output_folder)
    if args.para == 1:
        shutil.copy('./utils/params.yaml', output_folder)
    elif args.para == 2:
        shutil.copy('./utils/params_new.yaml', output_folder)
    
    logging.info(f"Number of PMDP states: {num_nodes * num_states}")
    logging.info(f"DRA: {num_nodes}")

    # Dijkstra Setup
    gridworld = create_gridworld_graph(grid_size, grid_size, rabin.coord_dict['obstacles'])
    logging.info("grid world setup successfully")


    # Transition Probability Table and Q-value
    nxa = np.zeros((num_states, num_actions))
    C = np.zeros((num_states, num_states, num_actions))
    TP = np.zeros((num_states, num_states, num_actions))
    nsa = np.zeros((num_states, num_nodes, num_actions)) # learning rate
    
    Q_values = np.zeros((num_states, num_nodes, num_actions))

    # init
    init_time = time.time()
    episode = 1
    time_counter = 1

    # Standard Q-Learning Algorithm 
    for episode in range(1, config['episodes']+1):
        
        if episode % config['save_epi'] == 0:
            logging.info(f"Saving episode: {episode}")
            np.save(os.path.join(output_folder, f"QTable_{episode}.npy"), Q_values)
        
        save_gap = config["save_gap"]
        cur_time_elapsed = time.time() - init_time
        if cur_time_elapsed >= save_gap * time_counter:
            logging.info(f"Saving at time: {int(cur_time_elapsed/60)} mins")
            np.save(os.path.join(output_folder, f'QTable_{int(cur_time_elapsed/60)}_mins.npy'), Q_values)
            time_counter += 1

        accept_counter = 0
        discount_episode_reward = 0
        
        state, node = init_state(num_states, num_nodes, rabin)
        mix_state = [state, node]

        reward_data = []
        delta_b = 0
        for step in range(1, config['max_steps']+1):
            ran = np.random.rand()
            # epsilon = 1 / (episode ** 0.1)
            epsilon = 1 / (episode ** 0.05) # big mdp

            if args.type == 1 or args.type == 4 or args.type == 5:
                delta_b = (1 - 1/(episode ** 0.15)) * epsilon # Figure 4(a) of journal 
            elif args.type == 2:
                # A = 0.00015
                A = config['our2_A'] # for big mdps
                if episode < 100:
                    g = 1 - 0.1 * math.exp(-A * episode)
                else:
                    g = 1 - 0.9 * math.exp(-A * episode)
                delta_b = (1 - g) * epsilon
            elif args.type == 3:
                # A = 0.0015
                A = config['our3_A'] # for big mdps
                if episode < 100:
                    g = 1 - 0.1 * math.exp(-A * episode)
                else:
                    g = 1 - 0.9 * math.exp(-A * episode)
                delta_b = (1 - g) * epsilon
            else:
                raise NotImplementedError

            if ran <= 1 - epsilon: # greedy
                action = np.argmax(Q_values[mix_state[0], mix_state[1]])
            else:
                if ran <= 1 - epsilon + delta_b: # biased 
                    action = bias_action(mix_state, grid_size, rabin, gridworld, TP)
                else:
                    action = np.random.randint(num_actions) # random explore
            
            next_state = take_action(mix_state[0], action, config)

            if args.type==4 or args.type == 5:
                if args.type == 4:
                    if episode > 30:
                        pass 
                    else:
                        C[state, next_state, action] += 1
                        nxa[state, action] += 1
                        TP[state, next_state, action] = C[state, next_state, action] / nxa[state, action]
                if args.type == 5:
                    if episode > 100:
                        pass
                    else:
                        C[state, next_state, action] += 1
                        nxa[state, action] += 1
                        TP[state, next_state, action] = C[state, next_state, action] / nxa[state, action]
            else:
                C[state, next_state, action] += 1
                nxa[state, action] += 1
                TP[state, next_state, action] = C[state, next_state, action] / nxa[state, action]

            # calculate rewards
            next_rabin_state, r, done, accept_counter = interpret(mix_state, next_state, rabin, accept_counter)
            discount_episode_reward += (discount_factor ** step) * r

            # update Q-value
            nsa[mix_state[0], mix_state[1], action] += 1
            max_next_q_value = np.max(Q_values[next_state, next_rabin_state])
            target_q_value = r + discount_factor * max_next_q_value
            Q_values[mix_state[0], mix_state[1], action] += 1/nsa[mix_state[0], mix_state[1], action] * (target_q_value - Q_values[mix_state[0], mix_state[1], action])
            mix_state = [next_state, next_rabin_state]
            state = next_state 
            if done == -1:
                break
        # write episode reward to csv
        reward_tmp = [episode, discount_episode_reward]
        reward_data.append(reward_tmp)
        with open(reward_file_name, 'a', newline='') as fd:
            csvwriter = csv.writer(fd)
            csvwriter.writerows(reward_data)

        writer.add_scalar("Train_Episode_Discount_Reward/train", torch.tensor(discount_episode_reward), episode)
        writer.flush() 
        epi_time = time.time() - init_time
        logging.info(f"Episode:{episode}|Time:{(epi_time):.1f} secs") 
        episode += 1
    
    logging.info(f"Time Elapsed: {((time.time() - init_time)):.2f} secs | total episode: {episode}")
    writer.close()
    np.save(os.path.join(output_folder, 'TP.npy'), TP)
    np.save(os.path.join(output_folder, 'QTable.npy'), Q_values)

        
if __name__ == '__main__':
    main()
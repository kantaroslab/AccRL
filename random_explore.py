import torch
import time
import csv
import shutil
import numpy as np
from utils.helper import * 
from utils.translate import *
import argparse

parser = argparse.ArgumentParser(description='Parameter')
parser.add_argument('--para', type=int, default=1)
args = parser.parse_args()

def main():
    clean_files(os.getcwd())
    # Load configuration & Setup output folder
    if args.para == 1:
        config = load_config('./utils/params.yaml')
    elif args.para == 2:
        config = load_config('./utils/params_new.yaml')
    discount_factor = config['discount_factor']
    logging, writer, output_folder = log_init(config, type='random')
    reward_file_name = os.path.join(output_folder, "discount_episode_reward.csv")
    
    # Define grid and action space 
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
        for step in range(1, config['max_steps']+1):
            ran = np.random.rand()
            epsilon = 1 / (episode ** 0.1)
            if ran <= 1 - epsilon: # greedy
                action = np.argmax(Q_values[mix_state[0], mix_state[1]])
            else:
                action = np.random.randint(num_actions) # random explore
            
            # calculate next MDP state
            next_state = take_action(mix_state[0], action, config)

            # update transition probability
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
        logging.info(f"Episode:{episode}|Time:{(epi_time ):.1f} secs") 
        episode += 1
    logging.info(f"Time Elapsed: {((time.time() - init_time)):.2f} secs | total episode: {episode}")
    writer.close()
    np.save(os.path.join(output_folder, 'TP.npy'), TP)
    np.save(os.path.join(output_folder, 'QTable.npy'), Q_values)


if __name__ == '__main__':
    main()
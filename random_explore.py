import torch
import time
import csv
import shutil
import numpy as np
from utils.helper import * 
from utils.translate import *
import argparse
from tqdm import tqdm
from utils.helper import * 
from utils.translate import *
from networkx.classes.digraph import DiGraph
from networkx import strongly_connected_components_recursive

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

    # Define grid and action space 
    grid_size = config['grid_size']
    num_states = config['grid_size'] * config['grid_size']
    num_actions = config['num_actions'] # left, right, up, down, stay idle

    # Transition probability (in practice)
    correct_prob = config['correct_prob']
    neighbor_prob = (1 - correct_prob) / (num_actions - 1)

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

    print(f"Training process completed. Now computing satisfaction prob")
    
    ###### Accepting Maximum Ending Components ######
    ###### Alg.47 P866 of Baier08 ######
    S = set()
    Sneg = set()
    acc_pmdp = set()
    successors = dict()
    props = dict()
    predecessors = dict()
    dlock = set()

    empirically = False  
    if not empirically:

        for mdp_state in range(num_states):
            for rabin_state in range(num_nodes):
                mix_state = [mdp_state, rabin_state]
                cur_pmdp = (mdp_state, rabin_state)
                if rabin_state in rabin.accept:
                    acc_pmdp.add(cur_pmdp) # Accept PMDP
                if rabin_state in rabin.deadlock:
                    dlock.add(cur_pmdp)
                if rabin_state not in rabin.reject:
                    Sneg.add(cur_pmdp) # PMDP/Accept
                S.add(cur_pmdp) # PMDP
                if cur_pmdp not in successors:
                    successors[cur_pmdp] = set()
                if cur_pmdp not in props:
                    props[cur_pmdp] = {}
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
                next_value_set = list(set(next_states.values()))
                for next_state_xy in next_value_set:
                    next_state = coordinates_to_index(next_state_xy[0], next_state_xy[1], grid_size)
                    next_rabin_state, _, _, _ = interpret(mix_state, next_state, rabin, 
                                                            accept_counter=0, is_baseline_collect=True)
                    next_pmdp = (next_state, next_rabin_state)
                    if (str(rabin_state), str(next_rabin_state)) in rabin.distance_map:
                        successors[cur_pmdp].add(next_pmdp)
                        if next_pmdp not in predecessors:
                            predecessors[next_pmdp] = set()
                        predecessors[next_pmdp].add(cur_pmdp)
                for next_pmdp in successors[cur_pmdp]:
                    if cur_pmdp != next_pmdp:
                        # if the next PMDP state is not the current one, 
                        # then it cannot be taken the action of 'stay idle' (4)
                        # because if we take stay idle, we stay at current position with prob 1
                        props[cur_pmdp][next_pmdp] = [v for v in range(num_actions-1)]
                    else:
                        props[cur_pmdp][next_pmdp] = [v for v in range(num_actions)]
        A = dict()
        actions = set()
        for a in range(num_actions):
            actions.add(a)
        for s in Sneg:
            # At any PMDP state, we can take all possible actions
            A[s] = actions
        print(f"Length of A: {len(A)}")
        MEC = set()
        MECnew = set()
        MECnew.add(frozenset(Sneg))
        k = 0
        print(f"init:{rabin.init_state} | acc: {rabin.accept} | rej: {rabin.reject}")
        while MEC != MECnew:
            k += 1
            MEC = MECnew
            MECnew = set()
            for T in MEC:
                R = set()
                T_temp = set(T)
                simple_digraph = DiGraph()
                for cur_pmdp in T_temp:
                    if cur_pmdp not in simple_digraph:
                        simple_digraph.add_node(cur_pmdp)
                    for next_pmdp in successors[cur_pmdp]:
                        if next_pmdp in T_temp:
                            simple_digraph.add_edge(cur_pmdp, next_pmdp)
                i = 0
                for Scc in strongly_connected_components_iterative(simple_digraph):
                    i += 1
                    if len(Scc) >= 1:
                        for s in Scc:
                            U_to_remove = set()
                            for u in A[s]:
                                for t in successors[s]:
                                    if (t not in Scc) and (u in props[s][t]):
                                        U_to_remove.add(u)
                            # print(U_to_remove)
                            A[s].difference_update(U_to_remove)
                            if not A[s]:
                                R.add(s)
                while R:
                    s = R.pop()
                    T_temp.remove(s)
                    for f in predecessors[s]:
                        if f in T_temp:
                            A[f].difference_udpate(set(props[f][s]))
                        if not A[f]:
                            R.add(f)
                j = 0
                for Scc in strongly_connected_components_iterative(simple_digraph):
                    j += 1
                    if len(Scc) >= 1:
                        common = set(Scc).intersection(T_temp)
                        if common:
                            MECnew.add(frozenset(common))
        AMECs = []
        for T in MEC:
            common = set(T.intersection(acc_pmdp))
            if common:
                for element in T:
                    AMECs.append(element)
        print(f"length of AMEC: {len(AMECs)}")

    pmax_iter = 30
    init_pmdp = []
    if empirically:
        for mdp_state in range(num_states):
            init_pmdp.append([mdp_state, rabin.init_state])

        ac = 0
        counter = 0
        for mix_state in tqdm(init_pmdp):
            accept_counter = 0
            counter += 1
            # print(f"start:{mix_state}")
            for step in range(1, config['max_steps']+1):

                action = np.argmax(Q_values[mix_state[0], mix_state[1]])

                next_state = take_action(mix_state[0], action, config)
                next_rabin_state, r, done, accept_counter = interpret(mix_state, next_state, rabin, accept_counter, is_baseline_collect=True)
                mix_state = [next_state, next_rabin_state]
                state = next_state 
                if next_rabin_state in rabin.accept:
                    ac += 1
                    break
                if done == -1:
                    break    
                if action == 8:
                    break 
        print(f"The satisfaction probability is {ac/counter}")
    else:
        p_max = np.zeros((num_states, num_nodes))
        for element in AMECs:
            p_max[element[0], element[1]] = 1

        # Calculate satisfaction prob using value iteration
        TP = {}
        # First, obtain the actual action for each of the PMDP state using the converged Q-table
        # Then, we the optimal policy (assuming the full knowledge of the TP) 
        # Define actual TP(* | s, a)
        optimal_policy = np.zeros((num_states, num_nodes))
        for state in range(num_states):
            x, y = index_to_coordinates(state, grid_size)
            for rabin_state in range(num_nodes):
                if (state, rabin_state) not in TP.keys():
                        TP[(state, rabin_state)] = {}
                action = np.argmax(Q_values[state, rabin_state]) #TODO: 
                optimal_policy[state, rabin_state] = action
                if action == 8:
                    TP[(state, rabin_state)][(state, rabin_state)] = 1
                else:
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
                    unchanged_action = []
                    for key, val in next_states.items():
                        if (x, y) == val:
                            unchanged_action.append(key)
                    if len(unchanged_action) > 0:
                        changed_action = [v for v in range(num_actions-1) if v not in unchanged_action]
                        for a in unchanged_action:
                            next_state = next_states[a]
                            next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                            mix_state = [state, rabin_state]
                            next_rabin_state, _, _, _ = interpret(mix_state, next_state, rabin, 0, is_baseline_collect=True)
                            prob = 0
                            if (str(rabin_state), str(next_rabin_state)) in rabin.distance_map:
                                if action in unchanged_action:
                                    prob = correct_prob + neighbor_prob * (len(unchanged_action)-1)
                                    TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
                                else:
                                    prob = neighbor_prob * len(unchanged_action)
                                    TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
                            else:
                                TP[(state, rabin_state)][(next_state, next_rabin_state)] = 0
                        for a in changed_action:
                            next_state = next_states[a]
                            next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                            mix_state = [state, rabin_state]
                            next_rabin_state, _, _, _ = interpret(mix_state, next_state, rabin, 0, is_baseline_collect=True)
                            prob = 0
                            if a == action:
                                prob = correct_prob
                            else:
                                prob = neighbor_prob
                            if (str(rabin_state), str(next_rabin_state)) in rabin.distance_map:
                                TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
                            else:
                                TP[(state, rabin_state)][(next_state, next_rabin_state)] = 0
                    else:
                        # all actions will lead to positional change
                        changed_action = [v for v in range(num_actions-1)]
                        for a in changed_action:
                            next_state = next_states[a]
                            next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                            mix_state = [state, rabin_state]
                            next_rabin_state, _, _, _ = interpret(mix_state, next_state, rabin, 0, is_baseline_collect=True)
                            prob = 0
                            if a == action:
                                prob = correct_prob
                            else:
                                prob = neighbor_prob
                            if (str(rabin_state), str(next_rabin_state)) in rabin.distance_map:
                                TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
                            else:
                                TP[(state, rabin_state)][(next_state, next_rabin_state)] = 0
        # Next, we compute the probs using value iteration (max)
        # https://www.cmi.ac.in/~madhavan/courses/qath-2015/slides/Lecture24.pdf 
        for _ in tqdm(range(pmax_iter)):
            for rabin_state in range(num_nodes):
                for mdp_state in range(num_states):
                    if (mdp_state, rabin_state) in AMECs:
                        continue
                    if rabin_state in rabin.deadlock:
                        continue 
                    tmp = 0
                    for next_state, prob in TP[(mdp_state, rabin_state)].items():
                        tmp += prob * p_max[next_state[0], next_state[1]]
                    p_max[mdp_state, rabin_state] = tmp
        result = []
        for mdp_state in range(num_states):
            result.append(p_max[mdp_state, rabin.init_state])
        
        print(f"Satisfaction Prob is {np.mean(result)}")




if __name__ == '__main__':
    main()
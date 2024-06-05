import os
import numpy as np
import time
import shutil
from tqdm import tqdm
from utils.helper import * 
from utils.translate import *
from networkx.classes.digraph import DiGraph
from networkx import strongly_connected_components_recursive

# https://www.youtube.com/watch?v=YLcUam8qfe4
# https://www.cmi.ac.in/~madhavan/courses/qath-2015/slides/Lecture24.pdf

def strongly_connected_components_iterative(graph):
    stack = []
    result = []
    visited = set()

    def dfs(node):
        component = []
        stack.append(node)
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                stack.extend([neighbor for neighbor in graph.neighbors(current) if neighbor not in visited])
        return component

    for node in graph.nodes():
        if node not in visited:
            result.append(dfs(node))

    return result
def main():
    clean_files(os.getcwd())
    config = load_config('./utils/params.yaml')
    discount_factor = config['discount_factor']
    runtime = config['runtime']
    logging, writer, output_folder = log_init(config, type='baseline')

    # Define grid and action space 
    grid_size = config['grid_size']
    num_states = config['grid_size'] * config['grid_size']
    num_actions = config['num_actions'] # left, right, up, down, stay idle

    # Initialize DRA 
    rabin = rabin_setup(config)
    num_nodes = rabin.num_of_nodes
    move2result(['command*'], output_folder)    
    shutil.copy('./utils/params.yaml', output_folder)

    # Transition probability (in practice)
    correct_prob = config['correct_prob']
    neighbor_prob = (1 - correct_prob) / (num_actions - 1)

    # print(f"{correct_prob}, {neighbor_prob}")

    # Product MDP Table Declaration
    V = np.zeros((num_states, num_nodes)) # Value function 
    optimal_policy = np.zeros((num_states, num_nodes))
    TP = np.zeros((num_states, num_nodes, num_actions, num_states, num_nodes))
    TP_MDP = np.zeros((num_states, num_actions, num_states))


    print(f"Size of TP: {num_states * num_nodes * num_actions * num_states * num_nodes}")
    for state in range(num_states):
        for rabin_state in range(num_nodes):
            x, y = index_to_coordinates(state, grid_size)
            TP[state, rabin_state, 8, state, rabin_state] = 1 # idle is applied deterministically
            TP_MDP[state, 8, state] = 1
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

            for candidate in range(num_actions - 1):
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
                        if candidate in unchanged_action:
                            prob = correct_prob + neighbor_prob * (len(unchanged_action)-1)
                            TP[state, rabin_state, candidate, next_state, next_rabin_state] = prob 
                            TP_MDP[state, candidate, next_state] = prob
                            # print(a, (x, y), '->', next_states[a],next_rabin_state, prob)
                        else:
                            prob = neighbor_prob * len(unchanged_action)
                            TP[state, rabin_state, candidate, next_state, next_rabin_state] = prob 
                            TP_MDP[state, candidate, next_state] = prob
                            # print(a, (x, y), '->', next_states[a],next_rabin_state, prob)
                    for a in changed_action:
                        next_state = next_states[a]
                        next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                        mix_state = [state, rabin_state]
                        next_rabin_state, _, _, _ = interpret(mix_state, next_state, rabin, 0, is_baseline_collect=True)
                        prob = 0
                        if a == candidate:
                            prob = correct_prob
                        else:
                            prob = neighbor_prob
                        TP[state, rabin_state, candidate, next_state, next_rabin_state] = prob 
                        TP_MDP[state, candidate, next_state] = prob
                else:
                    # all actions will lead to positional change
                    changed_action = [v for v in range(num_actions-1)]
                    for a in changed_action:
                        next_state = next_states[a]
                        next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                        mix_state = [state, rabin_state]
                        next_rabin_state, _, _, _ = interpret(mix_state, next_state, rabin, 0, is_baseline_collect=True)
                        prob = 0
                        if a == candidate:
                            prob = correct_prob
                        else:
                            prob = neighbor_prob
                        TP[state, rabin_state, candidate, next_state, next_rabin_state] = prob 
                        TP_MDP[state, candidate, next_state] = prob
    
    
    # C_emp = np.zeros((num_states, num_actions, num_states))
    # nxa_emp = np.zeros((num_states, num_actions))
    # TP_emp = np.zeros((num_states, num_actions, num_states))
    # logging.info(f"Collect empirical TP")
    # val = 1
    # tp_collect_start = time.time()
    # threshold  = 0.05
    # round = 0
    # flag = False
    # while not flag:
    #     if (time.time() - tp_collect_start)/60 > 60:
    #         break
    #     round += 1
    #     print(f"Round: {round}")
    #     for mdp_state in range(num_states):
            
    #         mix_state = [mdp_state, 0]
    #         accept_counter = 0
    #         action = np.random.randint(num_actions)
    #         next_state = take_action(mix_state[0], action, config)
    #         C_emp[mix_state[0],  action, next_state] += 1
    #         nxa_emp[mix_state[0],  action] += 1
    #         TP_emp[mix_state[0],  action, next_state] = C_emp[mix_state[0], action, next_state] / nxa_emp[mix_state[0],  action]
    #         mix_state = [next_state, 0]
    #         # if done != 0:
    #         #     break
    #         diff = np.abs(TP_emp - TP_MDP)
    #         val = np.max(diff)  
    #         # print(val)
    #         if val < threshold:
    #             flag = True 
    #             break
            
    #         if flag:
    #             break
    #     #$ print(f"difference: {val}")
    #     if flag:
    #         break
    #     print(f"difference: {val}")
    # logging.info(f"Time required for TP to be collected with difference of {threshold} is {time.time()-tp_collect_start} seconds")


    # Obtain reward for all possible PMDP state
    rewards = np.zeros((num_states, num_nodes, num_actions, num_states, num_nodes))
    for mdp_state in range(num_states):
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
        # create rabin list for 'current MDP state'
        if mdp_state not in rabin.coord_dict['obstacles']:
            rabin_list = [val for val in range(num_nodes) if val not in rabin.deadlock]
        else:
            rabin_list = [val for val in range(num_nodes) if val in rabin.deadlock]
        for rabin_state in rabin_list:
            mix_state = [mdp_state, rabin_state]
            for action, next_state in next_states.items():
                next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                next_rabin_state, r, _, _ = interpret(mix_state, next_state, rabin, 0, is_baseline_collect=True)
                rewards[mdp_state, rabin_state, action, next_state, next_rabin_state] = r

    # Value Iteration Main Function
    print(f"Running Value Iteration")
    convergence_threshold = 0.01
    init_time = time.time()
    num = 1
    while True:
        delta = 0
        for mdp_state in range(num_states):
            for rabin_state in range(num_nodes):
                Q_values = np.zeros(num_actions)
                mix_state = [mdp_state, rabin_state]
                for a in range(num_actions):
                    x, y = index_to_coordinates(mdp_state, grid_size)
                    expected_value = 0
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
                    for next_state in next_value_set:
                        next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                        for next_rabin_state in range(num_nodes):
                            expected_value += TP[mdp_state, rabin_state, a, next_state, next_rabin_state] * \
                                              (rewards[mdp_state, rabin_state, a, next_state, next_rabin_state] + \
                                                discount_factor * V[next_state, next_rabin_state])
                    Q_values[a] = expected_value
                new_value = np.max(Q_values)
                tmp = new_value - V[mdp_state, rabin_state]
                if tmp < 0:
                    tmp = -tmp 
                delta = max(delta, tmp)
                
                V[mdp_state, rabin_state] = new_value 
        epi_time = time.time() - init_time
        logging.info(f"Iteration: {num} | delta: {delta} |Time:{(epi_time / 60):.1f} min")
        num += 1
        if delta < convergence_threshold:
            break

    logging.info(f"Time Elapsed: {((time.time() - init_time) / 60):.2f} min")

    # Store the optimal policy using the converged V-function
    for mdp_state in range(num_states):
        for rabin_state in range(num_nodes):
            Q_values = np.zeros(num_actions)
            for a in range(num_actions):
                expected_value = 0 
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
                for next_state in next_value_set:
                    next_state = coordinates_to_index(next_state[0], next_state[1], grid_size)
                    for next_rabin_state in range(num_nodes):
                        expected_value += TP[mdp_state, rabin_state, a, next_state, next_rabin_state] * \
                                            (rewards[mdp_state, rabin_state, a, next_state, next_rabin_state] + \
                                              discount_factor * V[next_state, next_rabin_state])
                Q_values[a] = expected_value
            action = np.argmax(Q_values)
            optimal_policy[mdp_state, rabin_state] = action 
    np.save(os.path.join(output_folder, 'optimal_policy.npy'), optimal_policy)

    S = set()
    Sneg = set()
    acc_pmdp = set()
    successors = dict()
    props = dict()
    predecessors = dict()
    dlock = set()

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

    p_max = np.zeros((num_states, num_nodes)) # maximum probability for all states of PMDP
    for element in AMECs:
        p_max[element[0], element[1]] = 1

    not_reachable_states = []
    TP = {}
    for state in range(num_states):
        x, y = index_to_coordinates(state, grid_size)
        for rabin_state in range(num_nodes):
            if (state, rabin_state) not in TP.keys():
                    TP[(state, rabin_state)] = {}
            action = optimal_policy[state, rabin_state] # obtain action from the converged policy
            if action == 8:
                not_reachable_states.append((state, rabin_state))
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
                        if action in unchanged_action:
                            prob = correct_prob + neighbor_prob * (len(unchanged_action)-1)
                            TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
                        else:
                            prob = neighbor_prob * len(unchanged_action)
                            TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
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
                        TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
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
                        TP[(state, rabin_state)][(next_state, next_rabin_state)] = prob
    # Next, we compute the probs using value iteration (max)
    # https://www.cmi.ac.in/~madhavan/courses/qath-2015/slides/Lecture24.pdf

    for i in tqdm(range(50)):
        for mdp_state in range(num_states):
            for rabin_state in range(num_nodes):
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
        if mdp_state in rabin.coord_dict['obstacles']:
            continue 
        result.append(p_max[mdp_state, rabin.init_state])
    logging.info(f"average p_max: {np.mean(result)}")
        
if __name__ == '__main__':
    main()
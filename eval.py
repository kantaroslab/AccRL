from tqdm import tqdm 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from pulp import *
from utils.helper import *
matplotlib.use('Agg')
from utils.translate import *
from networkx.classes.digraph import DiGraph
from networkx import strongly_connected_components_recursive

# https://www.youtube.com/watch?v=YLcUam8qfe4
# https://www.cmi.ac.in/~madhavan/courses/qath-2015/slides/Lecture24.pdf


def smooth(scalars, weight=0.5):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

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
    # Load configuration & Setup output folder
    config = load_config('./utils/params.yaml')
    print(config, "\n")

    # Define grid and action space 
    grid_size = config['grid_size']
    num_states = config['grid_size'] * config['grid_size']
    num_actions = config['num_actions'] # left, right, up, down, stay idle

    # Transition probability (in practice)
    correct_prob = config['correct_prob']
    neighbor_prob = (1 - correct_prob) / (num_actions - 1)

    # Load Q-vals
    folder_name = config['eval_folder']
    print(f"Evaluation folder: {folder_name}")
    folders = list_subfolders(folder_name)

    ep    = [0]
    our1  = [0]
    our2  = [0]
    our3  = [0]
    our4 = [0]
    our5 = [0]
    ran   = [0]
    boltz = [0]
    ucb   = [0]

    rabin = rabin_setup(config)
    num_nodes = rabin.num_of_nodes
    print(f"deadlock: {rabin.deadlock}")

    """
    Accepting Maximum Ending Components
    Alg.47 P866 of Baier08
    """
    print(f"MDP state: {num_states} | DRA state: {num_nodes}")
    # Build S and Sneg
    S = set()
    Sneg = set()
    acc_pmdp = set()
    successors = dict()
    props = dict()
    predecessors = dict()
    dlock = set()

    empirically = True  
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
    print(f"Is calculating empirically? {empirically}")
    
    # for check_episode in [1000, 2000, 8000, 10000, 15000, 20000]:
    for check_episode in [10000, 30000, 50000, 80000, 100000]:
        ep.append(check_episode)
        init_pmdp = []
        for index, folder in enumerate(folders):

            if empirically:
                for mdp_state in range(num_states):
                    init_pmdp.append([mdp_state, rabin.init_state])

                Q_values = np.load(os.path.join(folder, f'QTable_{check_episode}.npy'))
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
                # print(f"{check_episode} | {folder.split('/')[-1].split('_')[0]} | Emp: {ac/counter}")
                print(f"{check_episode} | {folder} | Emp: {ac/counter}")
                if 'ours1' in folder:
                    our1.append(ac/counter)
                elif 'ours2' in folder:
                    our2.append(ac/counter)
                elif 'ours3' in folder:
                    our3.append(ac/counter)
                elif 'ours4' in folder:
                    our4.append(ac/counter)
                elif 'ours5' in folder:
                    our5.append(ac/counter)
                elif 'random' in folder:
                    ran.append(ac/counter)
                elif 'boltz' in folder:
                    boltz.append(ac/counter)
                elif 'ucb' in folder:
                    ucb.append(ac/counter)
                else:
                    pass
                continue
            else:
                p_max = np.zeros((num_states, num_nodes))
                for element in AMECs:
                    p_max[element[0], element[1]] = 1

                # Calculate satisfaction prob using value iteration
                TP = {}
                # First, obtain the actual action for each of the PMDP state using the converged Q-table
                # Then, we the optimal policy (assuming the full knowledge of the TP) 
                # Define actual TP(* | s, a)
                Q_values = np.load(os.path.join(folder, f'QTable_{check_episode}.npy'))
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
                
                print(f"{check_episode} | {folder.split('/')[-1].split('_')[0]} |  p_max: {np.mean(result)}")

                if 'ours1' in folder:
                    our1.append(np.mean(result))
                elif 'ours2' in folder:
                    our2.append(np.mean(result))
                elif 'ours3' in folder:
                    our3.append(np.mean(result))
                elif 'ours4' in folder:
                    our4.append(np.mean(result))
                elif 'ours5' in folder:
                    our5.append(np.mean(result))
                elif 'random' in folder:
                    ran.append(np.mean(result))
                elif 'boltz' in folder:
                    boltz.append(np.mean(result))
                elif 'ucb' in folder:
                    ucb.append(np.mean(result))
                else:
                    pass

    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Satisfaction Probability')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.plot()

    our1 = smooth(our1)
    our2 = smooth(our2)
    our3 = smooth(our3)
    our4 = smooth(our4)
    our5 = smooth(our5)
    ran = smooth(ran)
    boltz = smooth(boltz)
    ucb = smooth(ucb)

    plt.plot(ep, our1, "-k", ep, our2, "-r", ep, our3, "-b", ep, ran, "-g", ep, boltz, "-m", ep, ucb, "-c")
    plt.legend(["Biased 1", "Biased 2", "Biased 3", "Random", 'Boltzmann', 'UCB'], loc='best')

    # plt.plot(ep, our1, "-k", ep, our4, '#edb121', ep, our5, '#995318', ep, our2, "-r", ep, our3, "-b", ep, ran, "-g", ep, boltz, "-m", ep, ucb, "-c")
    # plt.legend(["Biased 1", "Biased 1-30", "Biased 1-100", "Biased 2", "Biased 3", "Random", 'Boltzmann', 'UCB'], loc='best')

    if empirically:
        plt.savefig(os.path.join(folder_name, 'empirical_result.png'))
    else:
        plt.savefig(os.path.join(folder_name, 'satisfaction_prob.png'))




if __name__ == '__main__':
    main()
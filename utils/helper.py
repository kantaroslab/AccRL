import os
import yaml
import glob
import shutil
import random
import numpy as np
import networkx as nx
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter


def clean_files(directory):
    files = os.listdir(directory)
    for file in files:
        if file.startswith('command'):
            os.remove(os.path.join(directory, file))


def init_state(num_states, num_nodes, rabin):
    state = np.random.randint(num_states)
    while state in rabin.coord_dict['obstacles']:
        state = np.random.randint(num_states)
    node = np.random.randint(num_nodes)
    while node in rabin.deadlock:
        node = np.random.randint(num_nodes)
    return state, node

def take_action(state, action, config):
    # Move the agent in the grid world upon each action taken
    # raise Exception("need to be modified")
    num_actions, grid_size = config['num_actions'], config['grid_size']
    correct_prob = config['correct_prob']
    neighbor_prob = (1 - correct_prob) / (num_actions - 1)

    if action == 8:
        # idle is applied deterministically
        return state 
    else:
        x, y = index_to_coordinates(state, grid_size)
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
        # print(x, y, next_states)
        
        candidates = {}
        unchanged_action = []
        for key, val in next_states.items():
            if (x, y) == val:
                unchanged_action.append(key)
        if len(unchanged_action) > 0:
            changed_action = [v for v in range(num_actions-1) if v not in unchanged_action]
            for act in unchanged_action:
                if action in unchanged_action:
                    candidates[next_states[act]] = correct_prob + neighbor_prob * (len(unchanged_action)-1)
                else:
                    candidates[next_states[act]] = neighbor_prob * len(unchanged_action)
            for act in changed_action:
                if act == action:
                    candidates[next_states[act]] = correct_prob
                else:
                    candidates[next_states[act]] = neighbor_prob
        else:
            if action == act:
                candidates[next_states[act]] = correct_prob
            else:
                candidates[next_states[act]] = neighbor_prob

        keys = list(candidates.keys())
        probabilities = list(candidates.values())
        chosen_key = random.choices(keys, weights=probabilities, k=1)[0]
        new_x, new_y = chosen_key 
        new_state = coordinates_to_index(new_x, new_y, grid_size)     
        return new_state 

def move2result(names=[], res_folder=None):
    for name in names:
        files = glob.glob(name)
        print(files)
        for file in files:
            if not os.path.isfile(os.path.join(res_folder, file)):
                shutil.move(file, res_folder)


def copy2result(names=[], res_folder=None):
    for name in names:
        files = glob.glob(name)
        for file in files:
            if not os.path.isfile(os.path.join(res_folder, file)):
                shutil.copy(file, res_folder)


def load_config(filename):
    with open(filename, "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

def check_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def log_init(config, type='ours', our_param_type=1):
    # Initialize logging for records
    now = datetime.now()
    current_time = now.strftime("%y%m%d_%H_%M_%S")
    if type == "ours":
        name = f"{type}{str(our_param_type)}_[" + f"{config['case_name'].replace(' ', '')}]_" + current_time
    elif type == "baseline" or type == "boltzmann" or type == "ucb" or type=='random':
        name = f"{type}_[" + f"{config['case_name'].replace(' ', '')}]_" + current_time
    else:
        raise NotImplementedError
    
    output_folder = os.path.join('./output', name)
    check_folder(output_folder)
    log_name = os.path.join(output_folder, "train.log")
    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(log_name), logging.StreamHandler()]
    # noinspection PyArgumentList
    logging.basicConfig(level=level, format=format, handlers=handlers)

    writer_dir = os.path.join(output_folder, "runs")
    writer = SummaryWriter(log_dir=writer_dir)
    return logging, writer, output_folder

### Grid World related

def coordinates_to_index(x, y, grid_size):
    # convert coordinates to state indices
    return x * grid_size + y

def index_to_coordinates(index, grid_size):
    # convert state indices to coordinates
    return index // grid_size, index % grid_size

def create_gridworld_graph(width, height, obstacles):
    """
    next_states = {
        0: (x, max(0, y - 1)), # left
        
        1: (x, min(grid_size - 1, y + 1)), # right 
        2: (max(0, x - 1), y), # up
        3: (min(grid_size - 1, x + 1), y), # down
        
        4: (max(0, x - 1), min(grid_size - 1, y + 1)), # up right
        5: (min(grid_size - 1, x + 1), max(0, y - 1)), # down left
        6: (max(0, x - 1), max(0, y - 1)), # up left,
        
        7: (min(grid_size - 1, x + 1), min(grid_size - 1, y + 1)), # down right
        8: (x, y), # unchanged
    }
    """
    grid_size = width
    G = nx.Graph()
    for i in range(width):
        for j in range(width):
            ind = coordinates_to_index(i, j, grid_size)
            l = (i, max(0, j - 1))
            r = (i, min(grid_size - 1, j + 1))
            u = (max(0, i - 1), j)
            d = (min(grid_size - 1, i + 1), j)

            ur = (max(0, i - 1), min(grid_size - 1, j + 1))
            dl = (min(grid_size - 1, i + 1), max(0, j - 1))
            ul = (max(0, i - 1), max(0, j - 1))
            dr = (min(grid_size - 1, i + 1), min(grid_size - 1, j + 1))
            if ind not in obstacles:
                if (i, j) != l and coordinates_to_index(l[0], l[1], grid_size) not in obstacles:
                    G.add_edge((i, j), l)
                if (i, j) != r and coordinates_to_index(r[0], r[1], grid_size) not in obstacles:
                    G.add_edge((i, j), r)
                if (i, j) != u and coordinates_to_index(u[0], u[1], grid_size) not in obstacles:
                    G.add_edge((i, j), l)
                if (i, j) != d and coordinates_to_index(d[0], d[1], grid_size) not in obstacles:
                    G.add_edge((i, j), d)
                if (i, j) != ur and coordinates_to_index(ur[0], ur[1], grid_size) not in obstacles:
                    G.add_edge((i, j), ur)
                if (i, j) != dl and coordinates_to_index(dl[0], dl[1], grid_size) not in obstacles:
                    G.add_edge((i, j), dl)
                if (i, j) != ul and coordinates_to_index(ul[0], ul[1], grid_size) not in obstacles:
                    G.add_edge((i, j), ul)
                if (i, j) != dr and coordinates_to_index(dr[0], dr[1], grid_size) not in obstacles:
                    G.add_edge((i, j), dr)
            
    return G

def dijkstra(grid_graph, start, end):
    return len(nx.shortest_path(grid_graph, source=start, target=end))

def heuristic(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance heuristic

def astar(grid_graph, start, end):
    return len(nx.astar_path(grid_graph, start, end, heuristic))

def list_subfolders(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    return subfolders

##### Choose Actions for Different Algorithms #####

def get_boltzmann_action(mix_state, Q_values, temperature, num_actions):
    mdp_state, dra_state = mix_state
    action_values = Q_values[mdp_state, dra_state, :]
    exp_values = np.exp(action_values / temperature)
    action_probs = exp_values / np.sum(exp_values)
    return np.random.choice(num_actions, p=action_probs)


def get_ucb_action(mix_state, Q_values, ucb_c, ns, nsa):
    mdp_state, dra_state = mix_state
    action_values = Q_values[mdp_state, dra_state, :]
    second_half = ucb_c * np.sqrt(2 * np.log(ns[mdp_state, dra_state] / nsa[mdp_state, dra_state]))
    return np.argmax(action_values + second_half)
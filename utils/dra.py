import os
import numpy as np
import collections
import heapq
import networkx as nx
import logging
import platform
import random


def check_sys2ap_rabin(state, coord_dict):
    result_ap = None
    is_hitting_obs = False
    for ap, element in coord_dict.items():
        if ap != 'obstacles':
            if int(state) == int(element):
                result_ap = ap
        else:
            for val in element:
                if int(state) == int(val):
                    result_ap = ap 
                    is_hitting_obs = True
    if is_hitting_obs:
        """
        If both non-obstacle AP and 'obstacles' are satisfied
        Then the obstacles will be satisfied prior to the goal
        But one should also check the position of the target
        """
        result_ap = "obstacles"
    return result_ap


def shortestPath(edges, source, sink):
    # https://gist.github.com/hanfang/89d38425699484cd3da80ca086d78129
    graph = collections.defaultdict(list)
    for l, r, c in edges:
        graph[l].append((c, r))
    # create a priority queue and hash set to store visited nodes
    queue, visited = [(0, source, [source])], set()
    heapq.heapify(queue)
    cnt = 0
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node not in visited:
            if cnt != 0:
                visited.add(node)
                path = path + [node]
                if node == sink:
                    return path
            cnt += 1
            for c, neighbour in graph[node]:
                if neighbour not in visited:
                    heapq.heappush(queue, (cost + c, neighbour, path))
    return None


class Rabin_Automaton_ROS(object):
    def __init__(self, config):
        self.ltl = config['ltl_task']

        # shortest paths for each dra state
        self.distance_map = {}
        # pruned dra_state - ap dictionary
        self.processed_dra = {}
        self.if_global = False
        
        with open("command_origin.ltl", mode='w') as f:
            f.write(self.ltl)
        if platform.system() == 'Darwin':
            # Macbook
            result1 = os.system("ltlfilt -l -F \"command_origin.ltl\" --output=\"command.ltl\"")
            result2 = os.system("./ltl2dstar --ltl2nba=spin:ltl2ba --output-format=dot command.ltl command.dot")
            result3 = os.system("ltlfilt --lbt-input -F \"command.ltl\" --output=\"command_read.ltl\"")
            result4 = os.system("./ltl2dstar command.ltl command.txt")
            
        else:
            result1 = os.system("./ltlfilt -l -F \"command_origin.ltl\" --output=\"command.ltl\"")
            result2 = os.system("./ltl2dstar --ltl2nba=spin:ltl2ba --output-format=dot command.ltl command.dot")
            result3 = os.system("./ltlfilt --lbt-input -F \"command.ltl\" --output=\"command_read.ltl\"")
            result4 = os.system("./ltl2dstar command.ltl command.txt")
            # result5 = os.system("dot -Tpdf command.dot > command.pdf")
        # Update: remove --stutter=no to make sure the translation is equal to HOA format for ./ltl2dstar
        if result1 == 0:
            logging.info("LTL Filtering Succeeded!")
        if result2 == 0:
            logging.info("DRA Dot Graph Generation Succeeded!")
        if result3 == 0:
            logging.info("LTL Translated into readable form in command_read.ltl")
        if result4 == 0:
            # Extract the number of accepting pairs information from the HOA file
            # Source: https://www.ltl2dstar.de/docs/ltl2dstar.html#output-format-hoa
            logging.info("HOA file generated successfully.")
            with open("command.txt") as f:
                lines = f.readlines()
            info = ""
            ap = ""
            for line in lines:
                line_ = line.replace("\n", "")
                if "Acceptance-Pairs" in line_:
                    info = int(line_.replace("Acceptance-Pairs: ", ""))
                if "AP" in line_:
                    ap = line_.replace('"', '').split()[2:]
            self.num_of_accepting_pairs = info
            self.ap = [element for element in ap]
            logging.info("Number of accepting pairs is: {}".format(self.num_of_accepting_pairs))

        # construct self.coord_dict
        self.coord_dict = {}
        self.ap_val = [] # get the AP values(numbers) for state initialization
        self.goals = []
        for element in self.ap:
            if not element.startswith("obstacles"):
                self.coord_dict[element] = int(element.replace('a', ''))
                self.ap_val.append(int(element.replace('a', '')))
                self.goals.append(int(element.replace('a', '')))

        if 'obstacles' in self.ap:
            self.coord_dict['obstacles'] = [int(element.replace('a', '')) for element in config['obstacle_list']]
            self.ap_val += [int(element.replace('a', '')) for element in config['obstacle_list']]
        else:
            self.coord_dict['obstacles'] = []

        rabin_graph = nx.nx_agraph.read_dot("command.dot")
        rabin_graph.remove_nodes_from(["comment", "type"])

        self.graph = rabin_graph
        self.num_of_nodes = len(self.graph.node)

        self.accept = [int(i) for i in self.graph.node if "+0" in self.graph.node[i]["label"]]
        self.reject = [int(i) for i in self.graph.node if "-0" in self.graph.node[i]["label"]]

        self.accepting_pair = {'L': self.accept, 'U': self.reject}

        self.deadlock = []
        for i in self.reject:
            if str(i) in self.graph[str(i)].keys():
                if " true" in [self.graph[str(i)][str(i)][j]["label"]
                               for j in range(len(self.graph[str(i)][str(i)]))]:
                    self.deadlock.append(i)
        
        for i in self.graph.node:
            if "fillcolor" in self.graph.node[i].keys():
                if self.graph.node[i]["fillcolor"] == "grey":
                    self.init_state = int(i)
                    break
        
        logging.info("initial: {}".format(self.get_init_state()))
        logging.info("accept: {}".format(self.accept))
        logging.info("reject: {}".format(self.reject))
        logging.info("Accepting pair: {}".format(self.accepting_pair))
        logging.info("deadlock: {}".format(self.deadlock))
        
        # print(self.accept, self.reject, self.init_state, self.ap)

    def get_graph(self):
        return self.graph

    def get_init_state(self):
        return self.init_state

    def check_current_ap(self, coord):
        # a system state can only be in 1 AP at a certain time step (correct)
        res_ap = []
        ans = check_sys2ap_rabin(coord, self.coord_dict)
        if ans is not None:
            res_ap.append(ans)
        # print("Atomic Proposition satisfied by {} -> {}".format(coord, res_ap))
        return res_ap

    def prune_dra(self, state_aps):
        # Edges will be pruned as long as there are more than 1 positive AP that need to be satisfied
        pruned_aps = []
        for ap in state_aps:
            pos, neg = seperate_ap_sentence(ap)
            if len(pos) > 1:
                continue
            pruned_aps.append(ap)
        return pruned_aps

    def dra_distance_dict_generation(self):
        # Set up the distance function which minimize the distance from current DRA state to the goal
        # since the DRA state is named based on increasing number
        # try to find the shortest path towards the biggest number
        # Data structure: (head, tail): {path}
        for start_node in self.graph.node:
            tmp_dict = {}
            for end_node in self.graph[start_node]:
                for k in range(len(self.graph[start_node][end_node])):
                    ap = self.graph[start_node][end_node][k]['label']
                    pos, _ = seperate_ap_sentence(ap)
                    if len(pos) <= 1:
                        if end_node not in tmp_dict:
                            tmp_dict[end_node] = []
                        tmp_dict[end_node].append(self.graph[start_node][end_node][k]['label'])
            self.processed_dra[start_node] = tmp_dict
        # print("DRA pruning completed")

        edges = []
        for start_node in self.processed_dra.keys():
            for end_node in self.processed_dra[start_node].keys():
                edges.append((start_node, end_node, 1))

        for start_node in self.graph.node:
            for end_node in self.graph.node:
                path = shortestPath(edges, start_node, end_node)
                if path is not None:
                    self.distance_map[(path[0], path[-1])] = path
        
        # if some reject DRA cannot reach accepting state, then treat them as deadlock as well
        connected_end_points = list(self.distance_map.keys())
        group = {}
        for tup in connected_end_points:
            key = tup[0]
            if key not in group:
                group[key] = []
            group[key].append(tup)
        for key, g in group.items():
            flag = False
            for tup in g:
                if int(tup[1]) in self.accept:
                    flag = True 
            if not flag:
                if int(tup[0]) not in self.deadlock:
                    self.deadlock.append(int(tup[0]))
    
                    
    def select_next_goal(self, source):
        # input is the current rabin state
        source_rabin = str(source[-1])
        target_rabin = None
        if self.accept == 1:
            target_rabin = str(self.accept[0])
            res_path = self.distance_map[(source_rabin, target_rabin)]
            if len(res_path) == 1:
                next_rabin = res_path[0]
            else:
                next_rabin = res_path[1]
        else:
            for target_rabin in self.accept:
                target_rabin = str(target_rabin)
                if (source_rabin, target_rabin) in self.distance_map:
                        res_path = self.distance_map[(source_rabin, target_rabin)]
                        if len(res_path) == 1:
                            next_rabin = res_path[0]
                        else:
                            next_rabin = res_path[1]
                        if next_rabin in self.processed_dra[source_rabin]:
                            break
        res_path = self.distance_map[(source_rabin, target_rabin)]
        new_goal_ap_sentence = self.processed_dra[source_rabin][next_rabin][0]
        pos, neg = seperate_ap_sentence(new_goal_ap_sentence)
        
        new_obstacles = [int(element.replace('a', '')) for element in neg if not element.startswith('obstacles')] 
        if len(pos) == 0 or (len(pos) == 1 and ' true' in pos[0]):
            closest_goal = None
            dis = float("inf")
            for goal in self.goals:
                if np.abs(goal - int(source[0])) < dis:
                    dis = np.abs(goal - int(source[0]))
                    closest_goal = goal 
            return closest_goal, new_obstacles
        # if len(pos) == 0 or (len(pos) == 1 and ' true' in pos[0]):
        #     return random.choice(self.goals), new_obstacles
        return self.coord_dict[str(pos[0]).strip()], new_obstacles

    def next_state(self, current_state, next_coord):
        ap_next = self.check_current_ap(next_coord)
        # print(ap_next)
        next_states = self.possible_states(current_state[-1])
        for i in next_states:
            next_state_aps = self.processed_dra[str(current_state[-1])][str(i)]
            if " true" in next_state_aps:
                return current_state[-1]
            else:
                for j in next_state_aps:
                    if self.check_ap(ap_next, j):
                        return i

    def possible_states(self, current_rabin_state):
        return [int(i) for i in self.processed_dra[str(current_rabin_state)].keys()]

    def check_ap(self, ap_next, ap_sentence):
        # print("ap_next->{} | ap_sentence->{}".format(ap_next, ap_sentence))
        pos, neg = seperate_ap_sentence(ap_sentence)
        # print("pos->{} | neg->{}".format(pos, neg))
        if set(ap_next).issuperset(set(pos)) and self.check_neg(ap_next, neg):
            """
            If ap_next satisfies:
                1) ap_next is the superset of the current positive APs in the sentence
                2) ap_next does not falls into any of the negative APs 
            Then we choose this ap_sentence as our next AP_sentence 
            """
            return True
        return False

    def check_neg(self, ap, negs):
        for i in ap:
            if i in negs:
                return False
        return True


def seperate_ap_sentence(input_str):
    return_str = []
    if len(input_str) > 1:
        index = find_ampersand(input_str)
        if len(index) >= 1:
            return_str = [input_str[0:index[0]]]
        else:
            return_str = input_str
            if '!' in return_str:
                return [], [return_str.replace('!', '')]
            else:
                return [return_str], []
        for i in range(1, len(index)):
            return_str += [input_str[index[i - 1] + 1:index[i]]]
        return_str = return_str + [input_str[index[-1] + 1:]]
        return_str = [i.replace(' ', '') for i in return_str]
    elif len(input_str) == 1:
        return_str = input_str
    elif len(input_str) == 0:
        raise AttributeError('input_str has no length')

    without_negs = []
    negations = []
    for i in range(len(return_str)):
        if '!' in return_str[i]:
            negations += [return_str[i].replace('!', '')]
        else:
            without_negs += [return_str[i]]
    return without_negs, negations


def find_ampersand(input_str):
    index = []
    original_length = len(input_str)
    original_str = input_str
    while input_str.find('&') >= 0:
        index += [input_str.index('&') - len(input_str) + original_length]
        input_str = original_str[index[-1] + 1:]
    return index

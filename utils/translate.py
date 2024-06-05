import logging
from utils.dra import Rabin_Automaton_ROS


def rabin_setup(config):
    rabin = Rabin_Automaton_ROS(config)
    rabin.dra_distance_dict_generation()
    return rabin


def rabin_dist(cur_rabin, rabin):
    # Check if the next rabin is closer to the goal
    source_rabin = cur_rabin
    target_rabin = None
    if len(rabin.accept) == 1:
        target_rabin = str(rabin.accept[0])  # assert accept rabin state is only 1
    else:
        for target_rabin in rabin.accept:
            target_rabin = str(target_rabin)
            if (source_rabin, target_rabin) in rabin.distance_map:
                res_path = rabin.distance_map[(source_rabin, target_rabin)]
                if len(res_path) == 1:
                    next_rabin = res_path[0]
                else:
                    next_rabin = res_path[1]
                if next_rabin in rabin.processed_dra[source_rabin]:
                    break
    res_path = rabin.distance_map[(source_rabin, target_rabin)]
    rabin_distance_to_goal = len(res_path) - 1  # do not count the distance to itself
    return rabin_distance_to_goal

def interpret(cur_mix_state, next_sys_state, rabin, accept_counter, is_baseline_collect=False):
    # setup interpretation function for the RL agent training process
    # such that it can detect which 'next rabin' state should the next system state belongs
    # based on the current rabin state
    next_rabin_state = rabin.next_state(cur_mix_state, next_sys_state)
    # Update: setup reward function according to the rabin automaton
    done = 0
    if next_rabin_state in rabin.accepting_pair['L'] and next_rabin_state not in rabin.accepting_pair['U']:
        # Update: check if the paths that start from L(accepting state) will go into U in the self.distance_map
        reward = 10
        accept_counter += 1
        if accept_counter >= 1:
            # end the current episode if the final rabin state has been arrived for 10 times
            # This is only needed when there is global requirement
            if not is_baseline_collect and accept_counter == 1:
                print("Accept rabin pair is reached, exiting..")
            done = 1
    else:
        if next_rabin_state in rabin.deadlock:
            # hits deadlock, no way to get out, exit current episode
            done = -1
            reward = -10
            # logging.info("Trapped in deadlock, exiting..")
        elif next_rabin_state in rabin.accepting_pair['U']:
            reward = -0.1
        else:
            reward = 0
    return next_rabin_state, reward, done, accept_counter
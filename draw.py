import csv 
import os
import matplotlib.pyplot as plt
import matplotlib 
from utils.helper import *
matplotlib.use('Agg')

def smooth(scalars, weight=0.9995):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def load_data(csv_name, folder=None):
    x, r = [], []
    # with open(os.path.join(f'./{folder}', csv_name, "discount_episode_reward.csv"), newline='') as csvfile:
    with open(os.path.join(csv_name, "discount_episode_reward.csv"), newline='') as csvfile:
        reader1 = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader1):
            # if i == 300000:
            #     break
            x.append(i)
            r.append(float(row[1]))
    r = smooth(r)
    return x, r

folder = 'taskE_50'
TITLE_NAME = f"Episode Reward"
SAVE_FIG_NAME = os.path.join(f'./{folder}', f"{folder}.png")
config = load_config('./utils/params.yaml')
folders = list_subfolders(folder)

for name in folders:
    if 'ours1' in name:
        CSV1 = name 
    if 'ours2' in name:
        CSV2 = name
    if 'ours3' in name:
        CSV3 = name
    if 'ours4' in name:
        CSV7 = name 
    if 'ours5' in name:
        CSV8 = name 
    if 'random' in name:
        CSV4 = name 
    if 'boltz' in name:
        CSV5 = name 
    if 'ucb' in name:
        CSV6 = name

plt.figure()
# plt.title()
plt.grid(True)

plt.xlabel('Episodes')
plt.ylabel('Accumulated Reward')

x1, r1 = load_data(CSV1)
x2, r2 = load_data(CSV2)
x3, r3 = load_data(CSV3)
x4, r4 = load_data(CSV4)
x5, r5 = load_data(CSV5)
x6, r6 = load_data(CSV6)

x7, r7, x8, r8 = None, None, None, None

time_bolt = "37.78"
time_our1 = "24.53"
time_our2 = "22.46"
time_our3 = "20.55"
time_rand = "27.31"
time_ucb  = "29.68"

time_our4 = "3.43"
time_our5 = "3.36"



b = 52550
o1 = 82710
o2 = 90140
o3 = -1
rd = 73570
uc = 67850

o4 = 43140
o5 = 44260

plt.plot(x1, r1, "-k", x2, r2, "-r", x3, r3, "-b", x4, r4, "-g", x5, r5, "-m", x6, r6, "-c", zorder=1)
plt.legend([f"Biased 1 ({time_our1} mins)", 
            f"Biased 2 ({time_our2} mins)", 
            f"Biased 3 ({time_our3} mins)", 
            f"Random ({time_rand} mins)", 
            f"Boltzmann ({time_bolt} mins)", 
            f"UCB ({time_ucb} mins)"], 
            loc='best')


# if 'taskG' in folder:
#     x7, r7 = load_data(CSV7)
#     x8, r8 = load_data(CSV8)
#     plt.plot(x1, r1, "-k", x7, r7, '#edb121', x8, r8, '#995318', x2, r2, "-r", x3, r3, "-b", x4, r4, "-g", x5, r5, "-m", x6, r6, "-c")
    # plt.legend([f"Biased 1 ({time_our1} mins)", 
    #             f"Biased 1-30 ({time_our4} mins)", 
    #             f"Biased 1-100 ({time_our5} mins)", 
    #             f"Biased 2 ({time_our2} mins)", 
    #             f"Biased 3 ({time_our3} mins)", 
    #             f"Random ({time_rand} mins)", 
    #             f'Boltzmann ({time_bolt} mins)', 
    #             f'UCB ({time_ucb} mins)'], 
    #             loc='upper left')
# else:
#     plt.plot(x1, r1, "-k", x2, r2, "-r", x3, r3, "-b", x4, r4, "-g", x5, r5, "-m", x6, r6, "-c", zorder=1)
#     plt.legend([f"Biased 1 ({time_our1} mins)", 
#                 f"Biased 2 ({time_our2} mins)", 
#                 f"Biased 3 ({time_our3} mins)", 
#                 f"Random ({time_rand} mins)", 
#                 f"Boltzmann ({time_bolt} mins)", 
#                 f"UCB ({time_ucb} mins)"], 
#                 loc='best')


# if 'taskG' in folder:
#     x7, r7 = load_data(CSV7)
#     x8, r8 = load_data(CSV8)
#     plt.plot(x1, r1, "-k", x7, r7, '#edb121', x8, r8, '#995318', x2, r2, "-r", x3, r3, "-b", x4, r4, "-g", x5, r5, "-m", x6, r6, "-c")
#     plt.legend([f"Biased 1 ({time_our1} min)", 
#                 f"Biased 1-30 ({time_our4} min)", 
#                 f"Biased 1-100 ({time_our5} min)", 
#                 f"Biased 2 ({time_our2} min)", 
#                 f"Biased 3 ({time_our3} min)", 
#                 f"Random ({time_rand} min)", 
#                 f'Boltzmann ({time_bolt} min)', 
#                 f'UCB ({time_ucb} min)'], 
#                 loc='upper left')
# else:
#     plt.plot(x1, r1, "-k", x2, r2, "-r", x3, r3, "-b", x4, r4, "-g", x5, r5, "-m", x6, r6, "-c", zorder=1)
#     plt.legend([f"Biased 1 ({time_our1} min)", 
#                 f"Biased 2 ({time_our2} min)", 
#                 f"Biased 3 ({time_our3} min)", 
#                 f"Random ({time_rand} min)", 
#                 f"Boltzmann ({time_bolt} min)", 
#                 f"UCB ({time_ucb} min)"], 
#                 loc='best')
    

plt.scatter(x5[b], r5[b], marker='*', color='black', s=80, zorder=3)
plt.scatter(x1[o1], r1[o1], marker='*', color='black', s=80, zorder=3)
plt.scatter(x2[o2], r2[o2], marker='*', color='black', s=80, zorder=3)
plt.scatter(x3[o3], r3[o3], marker='*', color='black', s=80, zorder=3)
plt.scatter(x4[rd], r4[rd], marker='*', color='black', s=80, zorder=3)
plt.scatter(x6[uc], r6[uc], marker='*', color='black', s=80, zorder=3)
# plt.scatter(x7[o4], r7[o4], marker='*', color='black', s=80, zorder=3)
# plt.scatter(x8[o5], r8[o5], marker='*', color='black', s=80, zorder=3)


plt.savefig(SAVE_FIG_NAME)
# plt.tight_layout()

"""
Script to check results of calculating dictionary of LCS lines across multiple points in time and space.
Uses results from flowfield.compute_ftle function, as demonstrated in DiscreteTurbulent_main.py.
Elle Stark February 2024
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Obtain LCS lines dictionary from pickle file
fname = 'data/LCS_data/alcs_backwardT0.6_spacing0.00025_0.6to10s_x200to800_y223to623.pickle'
with open(fname, 'rb') as handle:
    lcs_dict = pickle.load(handle)

# Plot LCS lines at a few times to check data
all_times = list(lcs_dict.keys())
select_times = [all_times[0], all_times[int(len(all_times)/4)], all_times[int(len(all_times)/2)],
                all_times[int(3*len(all_times)/4)], all_times[-1]]

# Loop through selected times and create line plot of all LCS lines for that time
for time in select_times:
    plt.close()
    fig, ax = plt.subplots()
    for i in range(len(lcs_dict[str(time)][0])):
        ax.plot(lcs_dict[str(time)][0][i], lcs_dict[str(time)][1][i], c='r',
                linewidth=0.5, linestyle="dashed")
    plt.show()




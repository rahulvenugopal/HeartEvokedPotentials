# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:25:54 2022
- Directory contains two folder: One for meditators, one for controls
- Starting point is this main directory
- Get the path + filenames of all .pkl files into two variables

- Select one sleep stage and pick .pkl files belonging to that stage
- Load the .pkl files in a loop and extract the chan*timepoints array
- Stack them as a 3D array chan*timepoints*subjects

- Transpose the dimensions for TFCE analysis format
- first dimension is the observations from each group (timepoints)
- last dimensions should be spatial (channels)
- 

@author: Rahul venugopal
"""
#%% Loading libraries
import mne
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test

#%% Load all the picked fils
data_dir_cnt = filedialog.askdirectory(title='Please select the directory with control subjects')
os.chdir(data_dir_cnt) # Changing directory to control folder

filelist_cnt = []

for root, dirs, files in os.walk(data_dir_cnt):
    for file in files:
        if file.endswith('.pkl'):
            #append the file name to the list if it is a .pkl file
            filelist_cnt.append(os.path.join(root,file))

data_dir_med = filedialog.askdirectory(title='Please select the directory with meditators')
os.chdir(data_dir_med) # Changing directory to control folder

filelist_med = []

for root, dirs, files in os.walk(data_dir_med):
    for file in files:
        if file.endswith('.pkl'):
            #append the file name to the list if it is a .pkl file
            filelist_med.append(os.path.join(root,file))

#%% Select relevant sleep stage .pkl files

sleep_stage_picker = 'REM'

# Select only REM files from the list
filelist_cnt_stage = [file for file in filelist_cnt if sleep_stage_picker in file]
filelist_med_stage = [file for file in filelist_med if sleep_stage_picker in file]

# Create 3D arrays subj * chans * timepoints

# control group
cnt_stage = []
for i in range(len(filelist_cnt_stage)):
    #create 3D arrays    
    cnt_stage.append(pd.read_pickle(filelist_cnt_stage[0])._data)

#stack all at once
all_controls = np.dstack(cnt_stage)

# meditators group
med_stage = []
for i in range(len(filelist_med_stage)):
    #create 3D arrays    
    med_stage.append(pd.read_pickle(filelist_med_stage[0])._data)

#stack all at once
all_meditators = np.dstack(med_stage)

#%% Gathering ingredients for TFCE analysis

X = [all_controls.transpose(1, 2, 0),
     all_meditators.transpose(1, 2, 0)]

tfce = dict(start=0, step=.2)  # ideally start and step would be smaller

# Load the mne info which was pickled that has channel info
info = pd.read_pickle('info.pkl')

# Get channel locs
montage = mne.channels.make_standard_montage('standard_1005')
info.set_montage(montage)

# Calculate adjacency matrix between sensors from their locations
adjacency, _ = find_ch_adjacency(info, "eeg")

# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(X, tfce,
                                                               n_jobs=-1,
                                                               adjacency=adjacency,
                                                               n_permutations=1000)

significant_points = cluster_pv.reshape(t_obs.shape).T < .05

print(str(significant_points.sum()) + " points selected by TFCE ...")

#%% Convert the arrays to evoked mne objects
all_controls_flipped = np.average(all_controls.transpose(2,0,1), axis = 0)
all_meditators_flipped = np.average(all_meditators.transpose(2,0,1), axis = 0)

# average across subjects
all_controls_mne = mne.io.RawArray(all_controls_flipped, info)
all_meditators_mne = mne.io.RawArray(all_meditators_flipped, info)

#%% Visualisation

# The results of these mass univariate analyses can be visualised by plotting
# :class:`mne.Evoked` objects as images (via :class:`mne.Evoked.plot_image`)
# and masking points for significance.
# Here, we group channels by Regions of Interest to facilitate localising
# effects on the head.

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([all_controls_mne,
                             all_meditators_mne],
                            weights=[1, -1])  # calculate difference wave

time_unit = dict(time_unit="ms")
evoked.plot_joint(title="Control vs. Meditators", ts_args=time_unit,
                  topomap_args=time_unit)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(info, midline="Cz")

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(axes=axes, group_by=selections, colorbar=False, show=False,
                  mask=significant_points, show_names="all", titles=None,
                  **time_unit)
plt.colorbar(axes["Left "].images[-1], ax=list(axes.values()), shrink=.3,
             label="µV")

plt.show()
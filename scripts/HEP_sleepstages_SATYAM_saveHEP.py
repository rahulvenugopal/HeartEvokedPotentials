# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:21:26 2025

Heartbeat Evoked Potentials (HEP) are EEG deflections time-locked to the
R-peak of the cardiac cycle.  They reflect the cortical processing of
interoceptive (heartbeat) signals and are modulated by arousal and sleep stage.

Pipeline overview
-----------------
1.  Load raw PSG (.edf) and pick EEG + ECG channels
2.  Detect R peaks from the ECG using NeuroKit2
3.  Embed the cleaned ECG and R-peak markers back into the MNE Raw object
4.  Filter, re-reference, and set electrode montage
5.  Load and upsample the scored hypnogram (YASA)
6.  Assign each R-peak to its sleep stage → metadata DataFrame
7.  Epoch the EEG around every R-peak; reject artefactual epochs
8.  Average epochs separately per sleep stage → HEP Evoked objects
9.  Plot and save figures + pickled Evokeds

@author: Rahul Venugopal
"""
#%% Loading libraries
import mne
import neurokit2 as nk
import numpy as np
import yasa
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

#%% Load PSG data

filename = 'data/TYBAUB_PSG.edf'
saver = os.path.splitext(filename)[0]

# preload=True loads all signal data into RAM immediately.
edfdata = mne.io.read_raw_edf(filename, preload = True)
srate = edfdata.info['sfreq']

# X4 and X5 are ECG channels, picking them and some EEG channels
# A1/A2 — mastoid reference electrodes (kept for re-referencing)
channels_to_pick = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4',
 'O1', 'O2', 'Cz', 'Pz','A1','A2', 'X4','X5']

eegrefchans = ['A1','A2']

# select only subset of channels
edfdata = edfdata.pick(channels_to_pick)

# Renaming ECG channels and setting them as ECG channels after filtering
mne.rename_channels(edfdata.info,
                    {'X4':'ECG1', 'X5':'ECG2'})

# ECG1 can be used for detecting R peaks
# Use edfdata.plot() to see the data

#%% Identifying R peaks using neurokit2

# Get the ECG data as 1D array
ecg_data = np.squeeze(edfdata.get_data(picks = ['ECG1']))

# Clean the ECG trace and see the data
signals, info1 = nk.ecg_process(ecg_data,
                               sampling_rate = edfdata.info['sfreq'])

# Find peaks
peaks, info2 = nk.ecg_peaks(signals["ECG_Clean"],
                           sampling_rate = edfdata.info['sfreq'])

# Pick R peak samples
locations = info2['ECG_R_Peaks']

# Get RR intervals
rr_intervals = np.diff(locations)
rr_seconds = rr_intervals / srate
rr_ms = rr_seconds * 1000

# Plot style
sns.set_style("whitegrid")
sns.set_context("talk")   # larger fonts

plt.figure(figsize=(6,1.5))
plt.boxplot(rr_ms, vert=False)
plt.axis("off")
plt.show()

#%% Adding cleaned ECG trace and identified ECG peak as a STIM

# creating new ECG cleaned channel from signals dataframe
cleaned_ECG = signals["ECG_Clean"].to_frame().transpose().to_numpy()

# create a mne raw object of clean ECG using info and raw object
info_ecg = mne.create_info(ch_names=['cleaned_ECG'],
                                    sfreq = edfdata.info['sfreq'],
                                    ch_types='ecg')
clean_ECG = mne.io.RawArray(cleaned_ECG, info_ecg)

# adding cleaned ECG channel to edfdata mne object
edfdata.add_channels([clean_ECG])

# creating a STIM channel based on identified R peaks (it has to be a 2D array)
flip_dataframe = peaks.transpose()
stim_data = flip_dataframe.to_numpy()

info_stim = mne.create_info(['STI'], edfdata.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(stim_data, info_stim)

# force_update_info should be True for STIM channel
edfdata.add_channels([stim_raw], force_update_info=True)

#%% The MNE objects should have same info features

# Then only we can add channels together
# Hence, filtering and re-referencing are done later

# filter, In MNE, filter applies only to EEG channels
edfdata.filter(l_freq=0.1, h_freq=40, fir_design='firwin')

# Setting channel type after filtering
edfdata.set_channel_types({'ECG1':'ecg', 'ECG2':'ecg'})

# Add channel locations from a template
montage = mne.channels.make_standard_montage('standard_1020')
edfdata.set_montage(montage)

# Re-referencing the data. The re-referenced info was preventing adding channels
# So, doing re-referencing at the very end
edfdata.set_eeg_reference(eegrefchans)
edfdata.drop_channels(eegrefchans)

# Dropping unwanted channels
edfdata.drop_channels('ECG2')

# Plot the sensors to make sure locations are intact
mne.viz.plot_sensors(edfdata.info)
plt.close()

# See the data now before epoching
edfdata.plot()

plt.close()

#%% Load EDF hypnogram read relevant details

# The hypnogram was scored by a sleep technologist and saved as an EDF
# annotations file.  Each annotation marks the start and duration of one
# 30-second sleep stage epoch.
sleep_data = mne.read_annotations('data/TYBAUB_HYPNO.edf')

# onset column is start of an epoch and duration tells us how long
hypnogram_annot = sleep_data.to_data_frame()

# Convert the 'duration' column from seconds → number of 30-second epochs.
# e.g. a 300-second block of N2 → duration = 300/30 = 10 epochs.
hypnogram_annot.duration = hypnogram_annot.duration/30

# Each description string looks like "Sleep Stage N2".
# We only need the last word (the stage code: W, N1, N2, N3, R).
just_labels = []
for entries in hypnogram_annot.description:
    just_labels.append(entries.split()[2])

# replacing the description column with just_labels
hypnogram_annot['description'] = just_labels

# Reconstruct a sample-by-sample hypnogram at 1 label per 30-second epoch.
# For each row in hypnogram_annot, repeat the stage label `duration` times.
# e.g. 10 epochs of N2  →  ['N2', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2']
hypno_30s = []
for stages in range(len(hypnogram_annot)):
    for repetitions in range(int(hypnogram_annot.duration[stages])):
        hypno_30s.append(hypnogram_annot.description[stages])

# convert list to NumPy array for YASA
hypno_30s = np.asarray(hypno_30s)

# YASA requires integer stage codes, not strings.
# hypno_str_to_int maps: 'W'→0, 'N1'→1, 'N2'→2, 'N3'→3, 'R'→4
hypno_30s = yasa.hypno_str_to_int(hypno_30s)

# Upsample from 1 label per 30 seconds to 1 label per sample.
# YASA tiles each 30-second epoch across (srate × 30) consecutive samples,
# giving an array of length == edfdata.n_times.
# sf_hypno = 1/30 means the hypnogram has one value every 30 seconds.
hypno_up_sampled = yasa.hypno_upsample_to_data(hypno = hypno_30s,
                            sf_hypno = 1/30,
                            data = edfdata)

# Print the sleep stages scored
print(np.unique(hypno_up_sampled))

#%% Creating metadata from locations and hypno_up_sampled

# MNE Epochs support a `metadata` argument — a DataFrame with one row per
# epoch.  We use it to tag each R-peak epoch with its sleep stage

'''
- Locations has datapoints where R peak happened
- Use this location as index and pick up the key in hypno_up_sampled
- This can go in loop and keep appending the sleep stage of that R peak
- This dataframe can go as metadata
'''

# For each detected R-peak, look up the sleep stage at that sample.
# hypno_up_sampled has one integer per sample of the recording.
# Indexing it with `locations` (array of R-peak sample positions) returns
# one stage integer per R-peak

# initialise a list for R peak's sleep stage
sleep_stage = []

for rpeaks in locations:
    sleep_stage.append(hypno_up_sampled[rpeaks])

# Converting list to a dataframe
sleep_stage_df = pd.DataFrame({'SleepStages':sleep_stage})
sleep_stage_df["SleepStages"].unique()

# Replacing the numerical code of sleep stages with strings
replacement_mapping_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "R"
}

# rewriting the SleepStages column
sleep_stage_series = sleep_stage_df["SleepStages"].replace(replacement_mapping_dict)

sleep_stage_df = sleep_stage_series.to_frame()
sleep_stage_df["SleepStages"].unique()

#%% Read events from annotations and epoch EDF data

# find_events() scans the STI channel for transitions from 0 to a non-zero value.
# Returns an array of shape (n_events, 3):
#   column 0: sample index of the event
#   column 1: previous channel value (usually 0)
#   column 2: event id (the non-zero value, here = 1)
events =  mne.find_events(edfdata, stim_channel='STI')

# Epoching parameters
tmin, tmax = -0.2, 1
baseline = (-0.2,0)

# rejecting bad epochs based on amplitude trheshold
reject_criteria = dict(eeg=250e-6) # 250 µV

flat_criteria = dict(eeg=1e-6) # 1 µV

# Epoch the data
epochs = mne.Epochs(edfdata,
                    events = events,
                    tmin = tmin,
                    tmax = tmax,
                    baseline = baseline,
                    detrend=1,
                    reject=reject_criteria,
                    flat=flat_criteria,
                    metadata = sleep_stage_df)

# Drop bad epochs
epochs.drop_bad()

# log count of bad epochs
print(epochs.drop_log)
epochs.plot_drop_log()

plt.savefig(filename + 'Log of bad epochs dropped.png',
            dpi = 600)

plt.close()

#%% Creating different epochs for each sleep stages

# Wake
stage = ['W']
W_epochs = epochs['SleepStages in {}'.format(stage)]
print('No of Wake epochs is ' + str(len(W_epochs.events)))
HEP_W = W_epochs.average()

# N1
stage = ['N1']
N1_epochs = epochs['SleepStages in {}'.format(stage)]
print('No of N1 epochs is ' + str(len(N1_epochs.events)))
HEP_N1 = N1_epochs.average()

# N2
stage = ['N2']
N2_epochs = epochs['SleepStages in {}'.format(stage)]
print('No of N2 epochs is ' + str(len(N2_epochs.events)))
HEP_N2 = N2_epochs.average()

# N3
stage = ['N3']
N3_epochs = epochs['SleepStages in {}'.format(stage)]
print('No of N3 epochs is ' + str(len(N3_epochs.events)))
HEP_N3 = N3_epochs.average()

# R
stage = ['R']
R_epochs = epochs['SleepStages in {}'.format(stage)]
print('No of R epochs is ' + str(len(R_epochs.events)))
HEP_R = R_epochs.average()

#%% Visualisation of Heart Evoked Potentials

channels_to_plot = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4',
 'O1', 'O2', 'Cz']

for channels in channels_to_plot:

    # For the actual visualisation, we store a number of shared parameters.
    style_plot = dict(
        colors={"W": "#d73027",
                "N1": "#fdae61",
                "N2": "#1a9850",
                "N3": "#3288bd",
                "R": "#01665e"},
        ci=.68, # CI is not showing up (or it is too tight a plot)
        show_sensors='lower right',
        truncate_yaxis="auto",
        picks=epochs.ch_names.index(channels))

    # Averaging each stage HEP and saving them to a dictionary
    evokeds = {"W":HEP_W,
               "N1":HEP_N1,
               "N2":HEP_N2,
               "N3":HEP_N3,
               "R":HEP_R}

    # Plotting the HEPs
    fig, ax = plt.subplots(figsize=(8, 4))
    mne.viz.plot_compare_evokeds(evokeds,
                                 axes=ax,
                                 **style_plot)

    plt.title("HEP in various stages of sleep from" + channels + ' sensor')
    plt.show()
    plt.tight_layout()

    # save the figure
    plt.savefig(saver +"_HEP_" + channels + '.png',
                dpi = 600)
    plt.close()

#%% Saving the epoched averaged files

with open(saver +'_W.pkl','wb') as f:
    pickle.dump(HEP_W,f)

with open(saver +'_N1.pkl','wb') as f:
    pickle.dump(HEP_N1,f)

with open(saver +'_N2.pkl','wb') as f:
    pickle.dump(HEP_N2,f)

with open(saver +'_N3.pkl','wb') as f:
    pickle.dump(HEP_N3,f)

with open(saver +'_REM.pkl','wb') as f:
    pickle.dump(HEP_R,f)

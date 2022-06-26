# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 13:49:56 2022

- Load sleep EDF data
- Select only one channel for final HEP
- Epoching done for all the channels
- Load scored hypnogram and upsample using YASA
- Run neurokit2's R peak algorithm
- Epoch the EEG data based on identified R peak
- Add sleep stages as metadata to the epochs

To DO
- HEPs show a clear ECG wave (P wave was obvious)
- Do epochs rejections
- Clean remaining epochs after rejection

@author: Rahul Venugopal
"""
#%% Loading libraries
import mne
import neurokit2 as nk
import numpy as np
import yasa
import pandas as pd
import matplotlib.pyplot as plt

#%% Load data and hypnogram files

edfdata = mne.io.read_raw_edf('Jyotiben_PSG.edf',
                             preload=True)

# X4 and X5 are ECG channels, picking them and say three EEG channels
channels_to_pick = ['Fz','Cz','Pz','A1','A2', 'X4','X5']
eegrefchans = ['A1','A2']

# select only subset of channels
edfdata.pick_channels(channels_to_pick)

# Re-referencing the data
edfdata.set_eeg_reference(eegrefchans)
edfdata.drop_channels(eegrefchans)

# Renaming ECG channels and setting them as ECG channels
mne.rename_channels(edfdata.info,
                    {'X4':'ECG1', 'X5':'ECG2'})
edfdata.set_channel_types({'ECG1':'ecg', 'ECG2':'ecg'})

# Add channel locations from a template
montage = mne.channels.make_standard_montage('standard_1020')
edfdata.set_montage(montage)

# Plot the sensors to make sure locations are intact
mne.viz.plot_sensors(edfdata.info)

# filter
edfdata.filter(0.1,None,fir_design='firwin').load_data()
edfdata.filter(None,40,fir_design='firwin').load_data()

edfdata.plot()

# ECG1 can be used for detecting R peaks

#%% Identifying R peaks using neurokit2

# Get the ECG data as 1D array
ecg_data = np.squeeze(edfdata.get_data(picks = ['ECG1']))

# Clean the ECG trace and see the data
signals, info1 = nk.ecg_process(ecg_data,
                               sampling_rate = edfdata.info['sfreq'])

# Find peaks
peaks, info2 = nk.ecg_peaks(signals["ECG_Clean"],
                           sampling_rate = edfdata.info['sfreq'])

# info2['ECG_R_Peaks'] contains the samplepoints of detected R peaks

#%% Load EDF hypnogram read relevant details

# load scored hypnogram file and read annotations neatly into a dataframe
sleep_data = mne.read_annotations('Jyotiben_PSG_scoredbygulshan.edf')

# onset column is start of an epoch and duration tells us how long
hypnogram_annot = sleep_data.to_data_frame()

# change the duration column into epochs count
hypnogram_annot.duration = hypnogram_annot.duration/30

# keep the description column neat
just_labels = []
for entries in hypnogram_annot.description:
    just_labels.append(entries.split()[2])

# replacing the description column with just_labels
hypnogram_annot['description'] = just_labels

# we need only the duration column and description column to recreate hypnogram
# just reapeat duration times the label in description column
hypno_30s = []
for stages in range(len(hypnogram_annot)):
    for repetitions in range(int(hypnogram_annot.duration[stages])):
        hypno_30s.append(hypnogram_annot.description[stages])

# converting list to numpy array
hypno_30s = np.asarray(hypno_30s)

# converting string array into int array using yasa
hypno_30s = yasa.hypno_str_to_int(hypno_30s)

hypno_up_sampled = yasa.hypno_upsample_to_data(hypno = hypno_30s,
                            sf_hypno = 1/30,
                            data = edfdata)

# Print the sleep stages scored
print(np.unique(hypno_up_sampled))

#%% Creating events out of R peaks samples and setup annotations

locations = info2['ECG_R_Peaks']

# adding two more columns to the events array
duration = np.squeeze(np.zeros((len(locations),1), dtype = 'int32'))
marker = np.squeeze(np.ones((len(locations),1), dtype = 'int32'))

# Events array in mne format
events_array = np.column_stack((locations, duration, marker))

# event dictionary with a label
event_dict_stim = {1 : 'R_Peaks'}

annot_from_events = mne.annotations_from_events(events = events_array,
                                                event_desc = event_dict_stim,
                                                sfreq = edfdata.info['sfreq'])
edfdata.set_annotations(annot_from_events)

#%% Creating metadata from locations and hypno_up_sampled
'''
- Locations has datapoint
- Use this location as index and pick up the key in hypno_up_sampled
- This can go in loop and keep appending the sleep stage of that R peak
- This dataframe can go as metadata
'''
# initialise a list for R peak's sleep stage
sleep_stage = []

for rpeaks in locations:
    sleep_stage.append(hypno_up_sampled[rpeaks])

# Converting list to a dataframe
sleep_stage_df = pd.DataFrame({'SleepStages':sleep_stage})

#%% Read events from annotations and epoch EDF data

events = np.array(mne.events_from_annotations(edfdata))

# Epoching parameters
tmin, tmax = -0.5, 1
baseline = None

# Epoch the data
epochs = mne.Epochs(edfdata,
                    events = events[0],
                    tmin = tmin,
                    tmax = tmax,
                    baseline = baseline,
                    detrend=1,
                    metadata = sleep_stage_df)

#%% Creating different epochs for each sleep stages

# Wake
stage = ['W']
W_epochs = epochs['SleepStages in {}'.format(stage)]
print('No of wake epochs is ' + str(len(W_epochs.events)))
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
    picks=epochs.ch_names.index("Fz"))

# Averaging each stage HEP and saving them to a dictionary
evokeds = {"W":HEP_W,
           "N1":HEP_N1,
           "N2":HEP_N2,
           "N3":HEP_N3,
           "R":HEP_R}

# Plotting the HEPs
fig, ax = plt.subplots(figsize=(6, 4))
mne.viz.plot_compare_evokeds(evokeds,
                             axes=ax,
                             **style_plot)

plt.title("HEP in various stages of sleep")
plt.tight_layout()
plt.show()

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
- Added detected ECG peaks as STIM channel
- Getting markers from the STIM channel
- Added cleaned ECG as a new channels
- Be mindful about inplace functions

Closed
- Do epochs rejections (completed on 29.07.2022)
Based on amplitide thresholds for EEG channel
- Auto save images

To DO


@author: Rahul Venugopal
"""
#%% Loading libraries
import mne
import neurokit2 as nk
import numpy as np
import yasa
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

#%% Load data and hypnogram files

edfdata = mne.io.read_raw_edf('Jyotiben_PSG.edf',
                             preload=True)

# X4 and X5 are ECG channels, picking them and say three EEG channels
channels_to_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz','A1','A2', 'X4','X5']
                    
eegrefchans = ['A1','A2']

# select only subset of channels
edfdata = edfdata.pick_channels(channels_to_pick)

# Renaming ECG channels and setting them as ECG channels after filtering
mne.rename_channels(edfdata.info,
                    {'X4':'ECG1', 'X5':'ECG2'})

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

# filter, In ME, filter applies only to EEG channels
edfdata.filter(0.1,None,fir_design='firwin').load_data()
edfdata.filter(None,40,fir_design='firwin').load_data()

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

#%% Creating metadata from locations and hypno_up_sampled
'''
- Locations has datapoints
- Use this location as index and pick up the key in hypno_up_sampled
- This can go in loop and keep appending the sleep stage of that R peak
- This dataframe can go as metadata
'''
# initialise a list for R peak's sleep stage
sleep_stage = []
locations = info2['ECG_R_Peaks']

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

events =  mne.find_events(edfdata, stim_channel='STI')

# Epoching parameters
tmin, tmax = -0.2, 1
baseline = (-0.2,0)

# rejecting bad epochs based on amplitude trheshold
reject_criteria = dict(eeg=250e-6) # 100 µV

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

plt.savefig('Log of bad epochs dropped.png',
            dpi = 600,
            backend='cairo')

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

channels_to_plot = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

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
    
    # save the figure
    plt.savefig("HEP from " + channels + ' sensor' + '.png',
                dpi = 600,
                backend='cairo')
    plt.close()
    
#%% Saving the epoched files as chan*times*epochs as .mat files

W_array = W_epochs.get_data()
W_array = W_array.transpose(1,2,0)

N1_array = N1_epochs.get_data()
N1_array = N1_array.transpose(1,2,0)

N2_array = N2_epochs.get_data()
N2_array = N2_array.transpose(1,2,0)

N3_array = N3_epochs.get_data()
N3_array = N3_array.transpose(1,2,0)

R_array = R_epochs.get_data()
R_array = R_array.transpose(1,2,0)

# saving to a mat file
scipy.io.savemat('W.mat', {'W_array': W_array})
scipy.io.savemat('N1.mat', {'N1_array': N1_array})
scipy.io.savemat('N2.mat', {'N2_array': N2_array})
scipy.io.savemat('N3.mat', {'N3_array': N3_array})
scipy.io.savemat('R.mat', {'R_array': R_array})
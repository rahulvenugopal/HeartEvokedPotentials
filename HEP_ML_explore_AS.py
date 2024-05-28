#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:56:21 2024

@author: arun
"""

import mne
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np

eegfilename = '/serverdata/ccshome/arun/NAS/Praerna/Controls/B03_Arun/B03_Arun_D1_audio1_05012023.vhdr'


raw = mne.io.read_raw_brainvision(eegfilename, preload=True) 
raw.resample(250)

# Get file info
srate       = raw.info.get('sfreq')    
chanlist    = raw.ch_names

crop_tmin = 30
crop_tmax = 150 

# Do Filtering
raw.filter(1.0,35,fir_design='firwin').load_data()

eeg = raw.copy().drop_channels(ch_names=['ECG1','ECG2']).crop(tmin=crop_tmin,tmax=crop_tmax)

# Add channel location
montage = mne.channels.make_standard_montage('standard_1005')
eeg.set_montage(montage)


#%% Compute ICA
from mne.preprocessing import ICA
ica1 = ICA(n_components=len(eeg.info.ch_names), max_iter='auto', random_state=99,
           method='infomax',fit_params=dict(extended=True),verbose=True)
ica1.fit(eeg)
ica1.plot_sources(eeg, show_scrollbars=False)
ica1.plot_components()

#%% Remove artefactual ICA
from mne_icalabel import label_components
ic_labels = label_components(eeg, ica1, method='iclabel')
labels = ic_labels["labels"]
preds = ic_labels["y_pred_proba"]
exclude_idx = [idx for idx, label in enumerate(labels) if label in ["eye blink","heart beat"] if preds[idx] > 0.9]
#'brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other'

reconst_eeg = eeg.copy()
ica1.apply(reconst_eeg.load_data(), exclude=exclude_idx)



#%% Detect ECG peaks and add as annotations
ecgdata = raw.copy().pick_channels(ch_names=['ECG1']).crop(tmin=crop_tmin,tmax=crop_tmax)._data[0]
# rpeaks, info = nk.ecg_peaks(ecgdata*1, sampling_rate=srate)
signals, vals     = nk.bio_process(ecgdata*1, sampling_rate=srate)


#%% Add ECG peaks as annotations to EEG data
ecgpeaktype = 'R'

# plt.figure(figsize=(20,8))
# plt.plot(ecgdata)
# plt.plot(np.where(rpeaks)[0],ecgdata[np.where(rpeaks)[0]],'ro')
# plt.plot(vals[f'ECG_{ecgpeaktype}_Peaks'],ecgdata[vals[f'ECG_{ecgpeaktype}_Peaks']],'yo')
# plt.plot(vals[f'ECG_{ecgpeaktype}_Peaks'],ecgdata[vals[f'ECG_{ecgpeaktype}_Peaks']],'ro')
# plt.xlim([0,0+srate*30])

n_events    = len(vals[f'ECG_{ecgpeaktype}_Peaks'])
onset       = np.array(vals[f'ECG_{ecgpeaktype}_Peaks'])/srate
duration    = np.repeat(0.1, n_events)
description = [f'{ecgpeaktype}_Peak'] * n_events
annotations = mne.Annotations(onset, duration, description)

#%% ICA-cleaned HEP
reconst_eeg.set_annotations(annotations)
events = mne.events_from_annotations(reconst_eeg)

icaepochs = mne.Epochs(reconst_eeg, events[0], event_id=events[1][f'{ecgpeaktype}_Peak'], tmin=-1, tmax=1)

icaerp = icaepochs.average()
icaerp.plot(ylim=dict(eeg=[-7, 15]),xlim=(-0.5, 1),
            titles=f'ICA cleaned | time-locked to {ecgpeaktype} peak')


#%% Non-ica HEP
nonicaeeg = raw.copy().drop_channels(ch_names=['ECG1','ECG2']).crop(tmin=crop_tmin,tmax=crop_tmax)
eeg.set_annotations(annotations)
events = mne.events_from_annotations(eeg)
nonicaepochs = mne.Epochs(eeg, events[0], event_id=events[1][f'{ecgpeaktype}_Peak'], tmin=-1, tmax=1)

nonicaerp = nonicaepochs.average()
nonicaerp.plot(ylim=dict(eeg=[-7, 15]),xlim=(-0.5, 1),
               titles=f'Non ICA cleaned | time-locked to {ecgpeaktype} peak')

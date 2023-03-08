#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEP analysis with surrogates

Created on Fri Feb 17 14:08:27 2023

@author: arun
"""

#%% Loading libraries
import mne
import neurokit2 as nk
import numpy as np
import yasa
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

#%% Define data paths
logbookfile = '/NAS_all/CCS_Common/SampleEEGData/Sleep_data/SATYAM_Logbook_updatedAnon_30122019.csv'
datadir     = '/NAS_all/CCS_Common/SampleEEGData/Sleep_data/AllEEGFiles'
scoringdir  = '/NAS_all/CCS_Common/SampleEEGData/Sleep_data/derivatives'

# Load Logbook
df = pd.read_csv(str(logbookfile), index_col=0)

# Get subject details
subj_no     = 32
filename    = df.at[subj_no,"MappingCode"]    
subjcode    = df.at[subj_no,"Subject_Code"]
sescode     = df.at[subj_no,"Session_ID"]

#%% Get the file paths
scoringfilename = (scoringdir + '/' + subjcode + '/' + sescode + 
                       '/hypnograms/' + filename + '.edf')
edffilename     = (datadir + '/' + subjcode + '/' + sescode + 
                       '/eeg/' + filename + '.edf')

#%% Load the edf file
edfdata = mne.io.read_raw_edf(edffilename, preload=True)

srate = edfdata.info['sfreq']

#%% Select only subset of channels

# X4 and X5 are ECG channels
channels_to_pick    = ['Fp1', 'Fp2', 'F3', 'F4', 'C3',  'Cz', 'C4', 'P3', 'P4','O1', 'O2',
                       'A1','A2', 'X4','X5']                   
eegrefchans         = ['A1','A2']

edfdata = edfdata.pick_channels(channels_to_pick)

# Renaming ECG channels and setting them as ECG channels after filtering
mne.rename_channels(edfdata.info,
                    {'X4':'ECG1', 'X5':'ECG2'})

edfdata.set_channel_types({'ECG1':'ecg', 'ECG2':'ecg'})
edfdata = edfdata.set_montage('standard_1005')


# ECG1 can be used for detecting R peaks

#%% Identifying R peaks using neurokit2

# Get the ECG data as 1D array
ecg_data = np.squeeze(edfdata.get_data(picks = ['ECG1']))

# Clean the ECG trace and see the data
signals, info1 = nk.ecg_process(ecg_data,sampling_rate = srate)

# Find peaks
peaks, info2 = nk.ecg_peaks(signals["ECG_Clean"],sampling_rate = srate)


stim_data = peaks.transpose().to_numpy()
stim_raw = mne.io.RawArray(stim_data, mne.create_info(['STI'], srate, ['stim']))
edfdata.add_channels([stim_raw], force_update_info=True)


#%% Read events from annotations and epoch EDF data
events = mne.find_events(edfdata, stim_channel='STI')
events = events[:1000] # Getting a subset for faster computation


#%% Epoching parameters
tmin, tmax = -0.2, 1
baseline = (-0.2,0)

#%% Original HEP
epochs = mne.Epochs(edfdata,
                    events = events,
                    tmin = tmin,
                    tmax = tmax,
                    baseline = baseline,
                    preload=False,
                    detrend=1)

HEP = epochs.average()
HEP_data = HEP.get_data()*1e6
epochtimes  = epochs._times_readonly
chanlist    = epochs.info.get('ch_names')

#%% Surrogate HEPs
n_surr = 50
surr_HEP_data = np.zeros([n_surr,HEP_data.shape[0],HEP_data.shape[1]])
for surr_no in range(n_surr):
    sur_events = events.copy()
    for i in range(len(sur_events)):
        sur_events[i,0] = sur_events[i,0] + (np.random.randint(int(srate*0.0),int(srate*0.4)) - int(srate*0.2))
        
    sur_epochs = mne.Epochs(edfdata,
                        events = sur_events,
                        tmin = tmin,
                        tmax = tmax,
                        baseline = baseline,
                        preload=False,
                        detrend=1)
    
    sur_HEP = sur_epochs.average()
    surr_HEP_data[surr_no] = sur_HEP.get_data()*1e6
    print('Computed Surrogate %i of %i' %(surr_no+1,n_surr))
    
    
#%% Plot HEPs with surrogate
chan_no = 13
plt.figure()
for surr_no in range(n_surr):
    plt.plot(epochtimes,surr_HEP_data[surr_no,chan_no,:],linewidth=0.35,color='silver')

plt.plot(epochtimes,HEP_data[chan_no,:],linewidth=1.5,color='black')
# plt.ylim([-5,5])
plt.title('%s' %(chanlist[chan_no]))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude ('r'$\mu$V)')
# plt.xlim([-0.2,epochtimes[-1]])
# plt.xticks(np.arange(-0.2,epochtimes[-1],0.2))



from scipy import stats
surr_CI_lo = np.zeros([HEP_data.shape[0],HEP_data.shape[1]])
surr_CI_hi = np.zeros([HEP_data.shape[0],HEP_data.shape[1]])
z_vals   = np.zeros([HEP_data.shape[0],HEP_data.shape[1]])
for chan_no in range(HEP_data.shape[0]):
    for timept in range(HEP_data.shape[1]):
        CI_val = stats.t.interval(alpha=0.95, loc=np.mean(surr_HEP_data[:,chan_no,timept]), 
                         df=len(surr_HEP_data[:,chan_no,timept])-1, 
                         scale=stats.sem(surr_HEP_data[:,chan_no,timept])) 
        surr_CI_lo[chan_no,timept] = CI_val[0]
        surr_CI_hi[chan_no,timept] = CI_val[1]
        z_vals[chan_no,timept]   = (HEP_data[chan_no,timept] - np.mean(surr_HEP_data[:,chan_no,timept]))/np.std(surr_HEP_data[:,chan_no,timept])

chan_no = 2
plt.figure()
plt.plot(epochtimes,HEP_data[chan_no,:],linewidth=1.5,color='black')
plt.plot(epochtimes,surr_CI_hi[0],linewidth=0.5,color='red')
plt.plot(epochtimes,surr_CI_lo[0],linewidth=0.5,color='blue')
plt.ylim([-5,5])
plt.title('%s' %(chanlist[chan_no]))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude ('r'$\mu$V)')


chan_no = 2
plt.figure()
plt.plot(epochtimes,HEP_data[chan_no,:],linewidth=0.5,color='silver')
plt.plot(epochtimes,z_vals[chan_no,:],linewidth=1.5,color='black')
# plt.ylim([-5,5])
plt.title('%s' %(chanlist[chan_no]))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude ('r'$\mu$V)')

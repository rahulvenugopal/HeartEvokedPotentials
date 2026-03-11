# -*- coding: utf-8 -*-
"""
HEP Surrogate & Pseudotrial Analysis
=====================================
Implements two control procedures for Heartbeat Evoked Potential (HEP) analysis,
following Steinfath et al. (2025), Imaging Neuroscience:
  "Validating genuine changes in heartbeat-evoked potentials using
   pseudotrials and surrogate procedures"

1. SURROGATE ANALYSIS
   - Shuffle the sequence of RR intervals to build new, fake R-peak timings.
   - The shuffled peaks preserve the *distribution* of inter-beat intervals
     but destroy the true phase-locking to the EEG.
   - Any HEP-like deflection surviving in surrogate data is NOT cardiac related.
   - Repeat N_SURROGATES times to build a null distribution.

2. PSEUDOTRIAL CORRECTION
   - For every real R-peak, place a pseudo-peak at ±50 % of the local RR
     interval (i.e. midway to the next beat), well away from any heartbeat.
   - Average pseudo-epochs captures heartbeat-INDEPENDENT EEG activity at
     a similar time scale.
   - Subtract pseudo-average from real HEP and the residual is the true HEP.

Inputs assumed to already exist (from your main pipeline):
    - `edfdata`        : filtered, re-referenced MNE Raw object
    - `locations`      : np.ndarray of R-peak sample indices (info2['ECG_R_Peaks'])
    - `rr_intervals`   : np.ndarray of RR intervals in SAMPLES
                         (np.diff(locations))
    - `sleep_stage_df` : metadata DataFrame with SleepStages per real epoch
    - `epochs`         : real HEP Epochs object (already epoched & bad-dropped)

Outputs
-------
    - surrogate_evokeds_per_stage : dict  {stage: list of N_SURROGATES Evoked}
    - pseudo_evokeds_per_stage    : dict  {stage: list of N_PSEUDOS Evoked}
    - corrected_hep_per_stage     : dict  {stage: Evoked}  (real - pseudo mean)

@author: adapted for SATYAM pipeline
"""

# ─── Libraries ────────────────────────────────────────────────────────────────
import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ─── User-facing parameters (edit these) ──────────────────────────────────────

N_SURROGATES   = 50      # how many surrogate iterations to run
N_PSEUDOS      = 50      # how many pseudotrial sets (each shifts all peaks once)

# Epoching window — keep identical to your main pipeline
TMIN, TMAX     = -0.2, 1.0
BASELINE       = (-0.2, 0.0)

# Rejection thresholds — keep identical to your main pipeline
REJECT         = dict(eeg=250e-6)
FLAT           = dict(eeg=1e-6)

# Sleep stages to loop over
STAGES         = ['W', 'N1', 'N2', 'N3', 'R']

# Path prefix for saving (mirrors `saver` in your main script)
SAVER          = 'data/TYBAUB_PSG'

# ─── NOTE ─────────────────────────────────────────────────────────────────────
# The block below shows how to derive `rr_intervals` from `locations`.
# In your main script, `locations = info2['ECG_R_Peaks']` is already available.
#
#   locations    = info2['ECG_R_Peaks']          # shape (n_peaks,)
#   rr_intervals = np.diff(locations)            # shape (n_peaks - 1,)
#
# Both variables are passed into the functions below.
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# HELPER : build a MNE STIM channel from an array of peak sample indices
# =============================================================================
def peaks_to_stim_raw(peak_indices, n_samples, sfreq):
    """
    Convert an array of sample indices into a binary MNE RawArray STIM channel.

    Parameters
    ----------
    peak_indices : array-like of int
        Sample positions of (real or surrogate/pseudo) R-peaks.
    n_samples : int
        Total number of samples in the recording (edfdata.n_times).
    sfreq : float
        Sampling frequency of the recording.

    Returns
    -------
    stim_raw : mne.io.RawArray
        Single-channel Raw object with 1 at each peak position, 0 elsewhere.
    """
    stim_signal = np.zeros((1, n_samples))

    # Clamp indices to valid range to avoid out-of-bounds writes
    valid = peak_indices[(peak_indices >= 0) & (peak_indices < n_samples)]
    stim_signal[0, valid.astype(int)] = 1

    info_stim = mne.create_info(['STI_SURR'], sfreq, ['stim'])
    return mne.io.RawArray(stim_signal, info_stim, verbose=False)


# =============================================================================
# SURROGATE ANALYSIS
# =============================================================================
def generate_surrogate_peaks(locations, rr_intervals, rng=None):
    """
    Create one set of surrogate R-peak sample indices by shuffling the order
    of observed RR intervals (Steinfath et al. 2025, surrogate procedure).

    The first R-peak anchor stays fixed so surrogate peaks remain inside the
    recording duration.  Only the sequence of inter-beat gaps is randomised,
    breaking phase-locking to EEG while preserving the RR distribution.

    Parameters
    ----------
    locations    : np.ndarray, shape (n_peaks,)
        Original R-peak sample positions.
    rr_intervals : np.ndarray, shape (n_peaks - 1,)
        Inter-beat intervals in samples (np.diff(locations)).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    surrogate_peaks : np.ndarray of int
        New R-peak positions derived from shuffled RR intervals.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Shuffle the RR sequence (in-place on a copy)
    shuffled_rr = rng.permutation(rr_intervals)

    # Reconstruct peak times: start from the original first peak, then
    # accumulate shuffled intervals
    surrogate_peaks = np.concatenate(([locations[0]],
                                       locations[0] + np.cumsum(shuffled_rr)))

    return surrogate_peaks.astype(int)


def run_surrogate_analysis(edfdata, locations, rr_intervals,
                           sleep_stage_df, n_iter=N_SURROGATES,
                           stages=STAGES):
    """
    Run the full surrogate analysis for N iterations and return per-stage
    averaged Evoked objects.

    For each iteration:
        1. Shuffle RR intervals → new surrogate peak positions
        2. Build a STIM channel, add temporarily to a Raw copy
        3. Epoch around surrogate peaks, attach the SAME metadata
           (sleep stage of the nearest real peak is re-used)
        4. Average per sleep stage → store Evoked

    Parameters
    ----------
    edfdata        : mne.io.Raw   (must NOT already have a STIM channel)
    locations      : np.ndarray   (real R-peak sample indices)
    rr_intervals   : np.ndarray   (np.diff(locations))
    sleep_stage_df : pd.DataFrame (SleepStages column, one row per real peak)
    n_iter         : int          (number of surrogate iterations)
    stages         : list of str  (sleep stage labels to collect)

    Returns
    -------
    surrogate_evokeds : dict  {stage_label: [Evoked, Evoked, ...]}
        Each list has `n_iter` entries (one Evoked per iteration).
    """
    surrogate_evokeds = {s: [] for s in stages}
    rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility

    n_samples = edfdata.n_times
    sfreq     = edfdata.info['sfreq']

    for i in range(n_iter):
        print(f'  Surrogate iteration {i+1}/{n_iter}')

        # 1. Generate shuffled peak positions
        surr_peaks = generate_surrogate_peaks(locations, rr_intervals, rng)

        # 2. Build temporary STIM channel and attach to a *copy* of the data
        stim_raw = peaks_to_stim_raw(surr_peaks, n_samples, sfreq)
        raw_copy = edfdata.copy().add_channels([stim_raw],
                                               force_update_info=True)

        # 3. Extract events from the temporary STIM channel
        events_surr = mne.find_events(raw_copy,
                                      stim_channel='STI_SURR',
                                      verbose=False)

        # The number of surrogate peaks may differ slightly from real peaks
        # (peaks near the end get clipped). Trim metadata to match.
        n_events = len(events_surr)
        meta_surr = sleep_stage_df.iloc[:n_events].reset_index(drop=True)

        # 4. Epoch around surrogate peaks
        epochs_surr = mne.Epochs(
            raw_copy,
            events=events_surr,
            tmin=TMIN, tmax=TMAX,
            baseline=BASELINE,
            detrend=1,
            reject=REJECT,
            flat=FLAT,
            metadata=meta_surr,
            preload=True,
            verbose=False
        )
        epochs_surr.drop_bad(verbose=False)

        # 5. Average per sleep stage and store
        for stage in stages:
            query = f'SleepStages in ["{stage}"]'
            try:
                stage_epochs = epochs_surr[f'SleepStages in ["{stage}"]']
                if len(stage_epochs) > 0:
                    surrogate_evokeds[stage].append(stage_epochs.average())
            except Exception:
                # Stage may be absent in this iteration — skip silently
                pass

        del raw_copy  # free memory

    return surrogate_evokeds


# =============================================================================
# PSEUDOTRIAL CORRECTION
# =============================================================================
def generate_pseudo_peaks(locations, rr_intervals, shift_fraction=0.5,
                           jitter_std_fraction=0.05, rng=None):
    """
    Generate pseudo R-peak positions for one pseudotrial set.

    Each pseudo-peak is placed at `shift_fraction` of the following RR interval
    after the corresponding real R-peak — i.e. midway to the next beat.
    A small random jitter (default ±5 % of RR) is added so that repeated
    pseudo sets are not identical.

    Why this works
    --------------
    The pseudo-peak lands at a point in the cardiac cycle that is maximally
    far from the true systolic / diastolic peaks, so it captures ongoing EEG
    that is NOT phase-locked to the heartbeat.  Subtracting the averaged
    pseudo-epoch from the real HEP removes the heartbeat-INDEPENDENT EEG
    component (Steinfath et al. 2025).

    Parameters
    ----------
    locations         : np.ndarray (n_peaks,)   real R-peak sample positions
    rr_intervals      : np.ndarray (n_peaks-1,) inter-beat intervals in samples
    shift_fraction    : float   fraction of RR interval to shift (default 0.5)
    jitter_std_fraction : float std of Gaussian jitter as fraction of RR
    rng               : np.random.Generator or None

    Returns
    -------
    pseudo_peaks : np.ndarray of int  (length n_peaks - 1)
        One pseudo-peak per real peak (except the last, which has no following
        RR interval).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Deterministic shift: fraction of each inter-beat interval
    shifts = (shift_fraction * rr_intervals).astype(float)

    # Add a small random jitter to vary each pseudotrial set
    jitter = rng.normal(loc=0.0,
                        scale=jitter_std_fraction * rr_intervals)
    shifts += jitter

    # Apply shifts to all peaks except the last (which has no RR_next)
    pseudo_peaks = locations[:-1] + shifts.astype(int)

    return pseudo_peaks.astype(int)


def run_pseudotrial_correction(edfdata, locations, rr_intervals,
                               sleep_stage_df, n_iter=N_PSEUDOS,
                               stages=STAGES):
    """
    Run pseudotrial correction for N iterations and return:
        - per-stage list of pseudo Evoked objects  (one per iteration)
        - per-stage corrected HEP  (real HEP − mean pseudo HEP)

    NOTE: `real HEP` is read back from the existing `epochs` object that was
    created in the main pipeline.  Pass `epochs` as a global or thread it
    through your calling code.

    Parameters
    ----------
    edfdata        : mne.io.Raw
    locations      : np.ndarray   (real R-peak sample indices)
    rr_intervals   : np.ndarray   (np.diff(locations))
    sleep_stage_df : pd.DataFrame (SleepStages, one row per real peak)
    n_iter         : int          (number of pseudotrial sets)
    stages         : list of str

    Returns
    -------
    pseudo_evokeds : dict  {stage: [Evoked, ...]}  length n_iter each
    """
    pseudo_evokeds = {s: [] for s in stages}
    rng = np.random.default_rng(seed=0)   # separate seed from surrogate RNG

    n_samples = edfdata.n_times
    sfreq     = edfdata.info['sfreq']

    for i in range(n_iter):
        print(f'  Pseudotrial iteration {i+1}/{n_iter}')

        # 1. Generate pseudo-peak positions for this iteration
        pseudo_peaks = generate_pseudo_peaks(locations, rr_intervals, rng=rng)

        # 2. Build STIM channel and attach to a copy of the raw data
        stim_raw = peaks_to_stim_raw(pseudo_peaks, n_samples, sfreq)
        raw_copy = edfdata.copy().add_channels([stim_raw],
                                               force_update_info=True)

        # 3. Find events
        events_pseudo = mne.find_events(raw_copy,
                                        stim_channel='STI_SURR',
                                        verbose=False)

        # Trim metadata to match number of pseudo events
        n_events = len(events_pseudo)
        meta_pseudo = sleep_stage_df.iloc[:n_events].reset_index(drop=True)

        # 4. Epoch around pseudo peaks
        epochs_pseudo = mne.Epochs(
            raw_copy,
            events=events_pseudo,
            tmin=TMIN, tmax=TMAX,
            baseline=BASELINE,
            detrend=1,
            reject=REJECT,
            flat=FLAT,
            metadata=meta_pseudo,
            preload=True,
            verbose=False
        )
        epochs_pseudo.drop_bad(verbose=False)

        # 5. Average per sleep stage and store
        for stage in stages:
            try:
                stage_epochs = epochs_pseudo[f'SleepStages in ["{stage}"]']
                if len(stage_epochs) > 0:
                    pseudo_evokeds[stage].append(stage_epochs.average())
            except Exception:
                pass

        del raw_copy

    return pseudo_evokeds


# =============================================================================
# CORRECTED HEP  (real − mean pseudo)
# =============================================================================
def compute_corrected_hep(real_evokeds_dict, pseudo_evokeds_dict):
    """
    Subtract the mean pseudotrial Evoked from the real Evoked for each stage.

    The mean across all pseudotrial iterations is taken first, then subtracted
    sample-by-sample from the real HEP.  This removes heartbeat-independent
    EEG fluctuations that contaminate the raw HEP.

    Parameters
    ----------
    real_evokeds_dict  : dict {stage: Evoked}  one real HEP per stage
    pseudo_evokeds_dict: dict {stage: [Evoked, ...]}  N pseudo HEPs per stage

    Returns
    -------
    corrected_hep : dict {stage: Evoked}  corrected HEP per stage
    """
    corrected_hep = {}

    for stage, real_evoked in real_evokeds_dict.items():
        pseudo_list = pseudo_evokeds_dict.get(stage, [])
        if not pseudo_list:
            print(f'  No pseudo evokeds for stage {stage}, skipping correction')
            corrected_hep[stage] = real_evoked
            continue

        # Average across all pseudotrial iterations
        mean_pseudo = mne.grand_average(pseudo_list)

        # Subtract: corrected = real − mean_pseudo
        corrected = real_evoked.copy()
        corrected.data = real_evoked.data - mean_pseudo.data
        corrected.comment = f'Corrected HEP {stage}'
        corrected_hep[stage] = corrected

    return corrected_hep


# =============================================================================
# VISUALISATION
# =============================================================================
def plot_surrogate_vs_real(real_evokeds_dict, surrogate_evokeds_dict,
                           channel, saver_prefix, stages=STAGES):
    """
    For each sleep stage, overlay the real HEP with all surrogate HEPs (grey)
    to visually assess whether the real HEP exceeds the surrogate envelope.

    Parameters
    ----------
    real_evokeds_dict     : dict {stage: Evoked}
    surrogate_evokeds_dict: dict {stage: [Evoked, ...]}
    channel               : str  channel name to plot (e.g. 'Cz')
    saver_prefix          : str  path prefix for saving figures
    stages                : list of str
    """
    stage_colors = {
        'W':  '#d73027',
        'N1': '#fdae61',
        'N2': '#1a9850',
        'N3': '#3288bd',
        'R':  '#01665e'
    }

    for stage in stages:
        real_evoked   = real_evokeds_dict.get(stage)
        surr_list     = surrogate_evokeds_dict.get(stage, [])

        if real_evoked is None or not surr_list:
            continue

        ch_idx = real_evoked.ch_names.index(channel)
        times  = real_evoked.times * 1000  # convert to ms

        fig, ax = plt.subplots(figsize=(9, 4))

        # Plot all surrogate traces in light grey
        for surr_evoked in surr_list:
            ax.plot(times, surr_evoked.data[ch_idx] * 1e6,
                    color='lightgrey', linewidth=0.6, alpha=0.7)

        # Overlay the real HEP in colour
        ax.plot(times, real_evoked.data[ch_idx] * 1e6,
                color=stage_colors.get(stage, 'black'),
                linewidth=2, label=f'Real HEP ({stage})')

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, label='R-peak')
        ax.axhline(0, color='k', linestyle='-',  linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Real vs Surrogate HEP — Stage {stage} — {channel}')
        ax.legend(loc='upper right')
        plt.tight_layout()

        fname = f'{saver_prefix}_surrogate_{stage}_{channel}.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f'  Saved → {fname}')


def plot_corrected_hep(real_evokeds_dict, corrected_hep_dict,
                       channel, saver_prefix, stages=STAGES):
    """
    For each sleep stage, plot raw HEP vs pseudotrial-corrected HEP side by
    side so the effect of correction is immediately visible.

    Parameters
    ----------
    real_evokeds_dict : dict {stage: Evoked}
    corrected_hep_dict: dict {stage: Evoked}
    channel           : str
    saver_prefix      : str
    stages            : list of str
    """
    stage_colors = {
        'W':  '#d73027',
        'N1': '#fdae61',
        'N2': '#1a9850',
        'N3': '#3288bd',
        'R':  '#01665e'
    }

    for stage in stages:
        real      = real_evokeds_dict.get(stage)
        corrected = corrected_hep_dict.get(stage)

        if real is None or corrected is None:
            continue

        ch_idx = real.ch_names.index(channel)
        times  = real.times * 1000

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        color = stage_colors.get(stage, 'black')

        axes[0].plot(times, real.data[ch_idx] * 1e6, color=color, linewidth=2)
        axes[0].axvline(0, color='k', linestyle='--', linewidth=0.8)
        axes[0].axhline(0, color='k', linewidth=0.5)
        axes[0].set_title(f'Raw HEP — {stage} — {channel}')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Amplitude (µV)')

        axes[1].plot(times, corrected.data[ch_idx] * 1e6,
                     color=color, linewidth=2, linestyle='--')
        axes[1].axvline(0, color='k', linestyle='--', linewidth=0.8)
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].set_title(f'Pseudotrial-Corrected HEP — {stage} — {channel}')
        axes[1].set_xlabel('Time (ms)')

        plt.suptitle(f'Pseudotrial correction — {channel}')
        plt.tight_layout()

        fname = f'{saver_prefix}_pseudocorrected_{stage}_{channel}.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f'  Saved → {fname}')


# =============================================================================
# SAVING
# =============================================================================
def save_surrogate_results(surrogate_evokeds, pseudo_evokeds,
                           corrected_hep, saver_prefix):
    """
    Pickle all outputs for later group-level analysis.

    Parameters
    ----------
    surrogate_evokeds : dict {stage: [Evoked, ...]}
    pseudo_evokeds    : dict {stage: [Evoked, ...]}
    corrected_hep     : dict {stage: Evoked}
    saver_prefix      : str
    """
    with open(saver_prefix + '_surrogate_evokeds.pkl', 'wb') as f:
        pickle.dump(surrogate_evokeds, f)
    print(f'Saved surrogate evokeds → {saver_prefix}_surrogate_evokeds.pkl')

    with open(saver_prefix + '_pseudo_evokeds.pkl', 'wb') as f:
        pickle.dump(pseudo_evokeds, f)
    print(f'Saved pseudotrial evokeds → {saver_prefix}_pseudo_evokeds.pkl')

    with open(saver_prefix + '_corrected_hep.pkl', 'wb') as f:
        pickle.dump(corrected_hep, f)
    print(f'Saved corrected HEP → {saver_prefix}_corrected_hep.pkl')


# =============================================================================
# MAIN — wire everything together
# =============================================================================
if __name__ == '__main__':

    # ── 0. Assume these exist from your main HEP pipeline ────────────────────
    #
    #   edfdata        : mne.io.Raw  (filtered, re-referenced, WITHOUT STI)
    #   locations      : np.ndarray  info2['ECG_R_Peaks']
    #   epochs         : mne.Epochs  (real HEP epochs, already drop-bad applied)
    #   sleep_stage_df : pd.DataFrame (SleepStages column)
    #
    # Derive RR intervals from the detected R-peaks
    rr_intervals = np.diff(locations)   # units: samples

    # ── 1. Build real HEP evokeds per stage (mirrors your main script) ────────
    real_evokeds = {}
    for stage in STAGES:
        stage_epochs = epochs[f'SleepStages in ["{stage}"]']
        if len(stage_epochs) > 0:
            real_evokeds[stage] = stage_epochs.average()
            print(f'Real HEP — {stage}: {len(stage_epochs)} epochs')

    # ── 2. Surrogate analysis ─────────────────────────────────────────────────
    print('\n=== Running surrogate analysis ===')
    surrogate_evokeds = run_surrogate_analysis(
        edfdata        = edfdata,
        locations      = locations,
        rr_intervals   = rr_intervals,
        sleep_stage_df = sleep_stage_df,
        n_iter         = N_SURROGATES,
        stages         = STAGES
    )

    # ── 3. Pseudotrial correction ─────────────────────────────────────────────
    print('\n=== Running pseudotrial correction ===')
    pseudo_evokeds = run_pseudotrial_correction(
        edfdata        = edfdata,
        locations      = locations,
        rr_intervals   = rr_intervals,
        sleep_stage_df = sleep_stage_df,
        n_iter         = N_PSEUDOS,
        stages         = STAGES
    )

    # ── 4. Compute corrected HEP (real − mean pseudo) ────────────────────────
    print('\n=== Computing corrected HEPs ===')
    corrected_hep = compute_corrected_hep(real_evokeds, pseudo_evokeds)

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    print('\n=== Plotting results ===')
    channels_to_plot = ['Cz', 'C3', 'C4', 'F3', 'F4']

    for ch in channels_to_plot:
        plot_surrogate_vs_real(real_evokeds, surrogate_evokeds,
                               channel=ch, saver_prefix=SAVER)
        plot_corrected_hep(real_evokeds, corrected_hep,
                           channel=ch, saver_prefix=SAVER)

    # ── 6. Save all outputs ───────────────────────────────────────────────────
    print('\n=== Saving results ===')
    save_surrogate_results(surrogate_evokeds, pseudo_evokeds,
                           corrected_hep, saver_prefix=SAVER)

    print('\nDone.')

import os
import mne
import sys

import numpy as np
from scipy.stats import entropy
import scipy.signal as signal
from sklearn.linear_model import LinearRegression


sys.path.append(os.path.abspath(os.getcwd()))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yasa


def read_hypno_and_plot(
    path: Path, plot: bool = False, figure_save_path: Path = None
) -> pd.DataFrame:
    """
    Reads a hypnogram from a tab-delimited text file, optionally plots the hypnogram,
    and returns the hypnogram as a pandas DataFrame.

    Parameters:
    -----
    path : Path
        The file path to the hypnogram text file.
    plot : bool, optional
        Whether to plot the hypnogram. Defaults to False.
    figure_save_path : Path, optional
        If plot is True, the path where the plot will be saved. If None,
        the plot will be saved in the same folder as the original data file. Defaults to None.

    Returns:
    --------
    pd.DataFrame:
        A DataFrame containing the hypnogram data, with columns for sleep stage, time from onset,
        epoch length, and a numerical code representing each sleep stage.
    """
    # Read the hypnogram file
    hypno = pd.read_csv(
        path,
        delimiter="\t",
        header=None,
        names=["stage", "time_from_onset", "epoch_length"],
    )

    # Map sleep stages to numerical codes
    mapper = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "L": -1}
    hypno["num"] = hypno["stage"].apply(lambda x: mapper[x])

    # Plot the hypnogram if requested
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5), constrained_layout=True)
        ax = yasa.plot_hypnogram(hypno["num"], fill_color="lightblue", ax=ax)

        # Save the figure to the specified path or default location
        if figure_save_path is None:
            fig.savefig(str(path).replace(".txt", "hypno.svg"))
        else:
            fig.savefig(figure_save_path)

    return hypno


def preprocess_eeg(
    raw: mne.io.Raw, epoch_length: float = 30.0, overlap: float = 0
) -> tuple:
    """
    Preprocess EEG data by applying band-pass filter, normalization, average referencing,
    and epoching into segments of specified length.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data to be preprocessed.
    epoch_length : float, optional
        Duration of each epoch (segment) in seconds (default is 30 seconds).
    overlap : float, optional
        Overlap duration between consecutive epochs (default is 0, meaning no overlap).

    Returns:
    --------
    tuple:
        epochs : mne.Epochs
            The preprocessed data split into epochs.
        means : np.ndarray
            Mean values of each channel across the full recording.
        stds : np.ndarray
            Standard deviation of each channel across the full recording.
        recording_time : float
            Total recording time in minutes.
    """
    # Apply filtering and drift removal
    raw = raw.load_data().filter(l_freq=0.5, h_freq=None, picks=["eeg", "ecg"])

    raw = raw.filter(l_freq=0.3, h_freq=None, picks="eog")

    raw = raw.filter(l_freq=10, h_freq=None, picks="emg")

    # Set average reference
    raw = raw.set_eeg_reference("average", projection=False)

    # Set bipolar references for non-EEG channels
    raw = mne.set_bipolar_reference(
        raw,
        anode=["RLEG+", "LLEG+", "CHEMG1", "ECG1"],
        cathode=["RLEG-", "LLEG-", "CHEMG2", "ECG2"],
        ch_name=["RLEG", "LLEG", "CHEMG", "ECG"],
        drop_refs=True,
    )
    raw = mne.set_bipolar_reference(
        raw, anode=["EOG1"], cathode=["EOG2"], ch_name=["EOGR"], drop_refs=False
    )

    # Apply filtering by signal type (EEG, EOG, EMG, ECG)
    raw = raw.filter(l_freq=None, h_freq=40, picks="eeg")
    raw = raw.filter(l_freq=None, h_freq=15, picks="eog")
    raw = raw.filter(l_freq=None, h_freq=50, picks="emg")
    raw = raw.filter(l_freq=None, h_freq=20, picks="ecg")
    # Normalize the data
    raw_data = raw.get_data()
    recording_time = raw_data.shape[1] / (
        raw.info["sfreq"] * 60
    )  # Total recording time in minutes
    means = raw_data.mean(axis=1, keepdims=True)
    stds = raw_data.std(axis=1, keepdims=True)
    # stds[stds == 0] = 1  # Prevent division by zero
    # normalized_data = (raw_data - means) / stds
    # raw._data = normalized_data

    # Epoch the data
    events = mne.make_fixed_length_events(raw, duration=epoch_length, overlap=overlap)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_length, baseline=None, detrend=None
    )

    return epochs, means, stds, recording_time, events


def extract_spectral_features(
    psd: np.array, freqs: np.array, ch_names: np.array, band_dict: dict[str:tuple]
) -> dict:
    """
    Extract spectral features from an epoch, including power in frequency bands, log transformations,
    spectral slope, and entropy measures.

    Parameters:
    -----------
    epoch_spectrum : mne.time_frequency.EpochsTFR
        Power spectral density (PSD) for a single epoch.
    band_dict : a dictionary with the band names as key and values with a tuple

    Returns:
    --------
    dict:
        Dictionary containing the extracted features for each channel and frequency band.
    """

    features = {}

    for i, channel in enumerate(ch_names):  # epoch_spectrum.ch_names:
        # psd = epoch_spectrum.get_data(picks=channel)
        total_power = psd[i].sum()
        if total_power == 0:
            psd = np.ones_like(psd)
            total_power = psd.sum()

        normalized_psd = psd[i] / total_power
        low_non_zero_value = np.min(normalized_psd[np.nonzero(normalized_psd)])
        log_psd = np.log(normalized_psd * (np.exp(1) / low_non_zero_value)) # get a stable relative log psd

        log_total_power = log_psd.sum()
        normalized_log_psd = log_psd / log_total_power

        features[f"{channel}_root_total_power"] = np.sqrt(total_power)
        features[f"{channel}_spectral_entropy"] = entropy(normalized_psd)

        for band_name, (fmin, fmax) in band_dict.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = normalized_psd[idx_band].sum()
            log_band_power = normalized_log_psd[idx_band].sum()
            features[f"{channel}_{band_name}_power"] = band_power
            features[f"{channel}_log_{band_name}_power"] = log_band_power

        freqs_reshaped = np.log(freqs.reshape(-1, 1))
        lin_reg_log = LinearRegression().fit(
            freqs_reshaped, len(freqs_reshaped) * normalized_log_psd
        )

        features[f"{channel}_log_spectral_slope"] = lin_reg_log.coef_[0]
        features[f"{channel}_log_spectral_intercept"] = lin_reg_log.intercept_

    return features


def extract_eog_features(eog_epoch_spectrum: mne.time_frequency.EpochsTFR) -> dict:
    """
    Extract EOG features including power in horizontal and vertical channels.

    Parameters:
    -----------
    eog_epoch_spectrum : mne.time_frequency.EpochsTFR
        EOG power spectral density for a single epoch.

    Returns:
    --------
    dict:
        Dictionary containing extracted EOG features for each channel.
    """
    features = {}
    mapper = {
        "EOG1": "vertical_left",
        "EOG2": "horizontal_right",
        "EOGR": "r_l_diff",
    }

    for channel in eog_epoch_spectrum.ch_names:
        psd = eog_epoch_spectrum.get_data(picks=channel)
        features[f"EOG_{mapper[channel]}"] = psd[0, 0, :].sum()

    # TOTEST: it could be that the relative and not abs power would be more informative here,
    # in any way this should be refined and eog channel mey be better treated as additional EEG and extraction of the eye component may be optimal for the evaluation of cleaner eye moments
    return features


def extract_ecg_features(ecg_epoch_data: mne.Epochs) -> dict:
    """
    Extract ECG features, including heart rate (HR), standard deviation of normal-to-normal intervals (SDNN),
    and root mean square of successive differences (RMSSD) between intervals.

    Parameters:
    -----------
    ecg_epoch_data : mne.Epochs
        ECG epoch data.

    Returns:
    --------
    dict:
        Dictionary containing extracted ECG features (HR, SDNN, RMSSD).
    """
    features = {}
    refractory_period = 60 / 220  # Minimum beat interval in seconds for 220 bpm
    sample_frequency = ecg_epoch_data.info["sfreq"]

    for channel in ecg_epoch_data.info["ch_names"]:
        ecg = ecg_epoch_data.get_data()
        squared_signal = np.diff(ecg[0, 0, :]) ** 2  # Emphasize peaks
        peaks, __annotations__ = signal.find_peaks(
            squared_signal,
            height=(np.mean(squared_signal), None),
            distance=int(refractory_period * sample_frequency),
        )

        rr_intervals = np.diff(peaks) / sample_frequency
        norm_rr_intervals = rr_intervals / np.mean(rr_intervals)

        features[f"{channel}_heart_rate"] = 60 / np.mean(rr_intervals)
        features[f"{channel}_SDNN"] = np.std(norm_rr_intervals)
        features[f"{channel}_rmssd"] = np.sqrt(np.mean(np.diff(norm_rr_intervals) ** 2))

    return features


def extract_emg_features(emg_epoch_spectrum: mne.time_frequency.EpochsTFR) -> dict:
    """
    Extract EMG features, specifically power in a specified frequency band.

    Parameters:
    -----------
    emg_epoch_spectrum : mne.time_frequency.EpochsTFR
        EMG power spectral density for a single epoch.

    Returns:
    --------
    dict:
        Dictionary containing extracted EMG features.
    """
    features = {}
    band_dict = {"power": (10, 50)}
    freqs = emg_epoch_spectrum.freqs

    for channel in emg_epoch_spectrum.ch_names:
        psd = emg_epoch_spectrum.get_data(picks=channel)
        for band_name, (fmin, fmax) in band_dict.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            features[f"{channel}_{band_name}"] = psd[0, 0, idx_band].sum()

    return features

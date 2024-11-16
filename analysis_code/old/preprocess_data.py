"This file is used to load and preprocess the EEG data to a labeled feature matrix"

import os
import mne
import glob
import sys
from pathlib import Path
import numpy as np

from scipy.stats import entropy
import scipy.signal as signal
from sklearn.linear_model import LinearRegression


import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.getcwd()))
from analysis_code.hypnogram_read import read_hypno_and_plot


def preprocess_eeg(raw, epoch_length: float = 30.0, overlap: float = 0):
    """
    Preprocess EEG data by applying band-pass filter, normalization,
    average reference, and epoching into 30-second segments.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data to preprocess.

    Returns:
    epochs : mne.Epochs
        The preprocessed data as epochs.
    """
    # 3. low pass filtering:

    # 1. filter the data to for drift in EEG and ECG
    # Remove the drift from all signals
    raw = raw.load_data().filter(
        l_freq=0.5, h_freq=None, picks=["eeg", "ecg", "eog"], n_jobs=1
    )

    raw = raw.copy().filter(
        l_freq=10, h_freq=None, picks="emg", n_jobs=1
    )  # remove the drift and movement patterns keeping the twitching information
    # EEG + ECG

    # Set an average reference
    raw = raw.set_eeg_reference("average", projection=False)

    # set bipolar ref to non EEG channels
    raw = mne.set_bipolar_reference(
        raw,
        anode=["RLEG+", "LLEG+", "CHEMG2", "ECG1"],
        cathode=["RLEG-", "LLEG-", "CHEMG1", "ECG2"],
        ch_name=["RLEG", "LLEG", "CHEMG", "ECG"],
        drop_refs=True,
    )

    raw = mne.set_bipolar_reference(
        raw,
        anode=["EOG1"],
        cathode=["EOG2"],
        ch_name=["EOGR"],
        drop_refs=False,
    )

    # EEG + ECG
    raw = raw.filter(l_freq=None, h_freq=40, picks="eeg", n_jobs=1)
    # EOG
    raw = raw.filter(l_freq=None, h_freq=15, picks="eog", n_jobs=1)
    # EMG - filter only the line noise
    raw = raw.copy().filter(
        l_freq=None, h_freq=50, picks="emg", n_jobs=1
    )  # keep muscle twitching basic frequency
    # ECG - filter only the line noise
    raw = raw.filter(l_freq=None, h_freq=20, picks="ecg", n_jobs=1)

    # 4. Z-score normalization of the channels over the compleat night
    # Normalize each channel by subtracting the mean and dividing by standard deviation
    raw_data = raw.get_data()
    recording_time = raw_data.shape[1] / (raw.info["sfreq"] * 60)  # in minutes

    means = raw_data.mean(axis=1, keepdims=True)
    stds = raw_data.std(
        axis=1, keepdims=True
    )  # Get data as numpy array (channels x time)
    stds[stds == 0] = 1  # Prevent division by zero in case any channel is flat
    normalized_data = (raw_data - means) / stds
    raw._data = normalized_data  # Set the normalized data back to raw object

    # 5. Epoch the data into consecutive 30-second segments
    events = mne.make_fixed_length_events(raw, duration=epoch_length, overlap=overlap)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_length, baseline=None, detrend=None
    )

    return epochs, means, stds, recording_time


def extract_spectral_features(epoch_spectrum):
    """
    Extract various features from each epoch such as power in frequency bands, log transformations,
    linear fits, and entropy.

    Parameters:
    epoch : mne.Epochs
        Single epoch of data from which features are to be extracted.

    Returns:
    dict : A dictionary containing the features for each channel and frequency band.
    """

    """a function to apply on each epoch"""

    band_dict = {
        "low_delta": (0.5, 1.5),
        "high_delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "low_sigma": (10, 12),
        "high_sigma": (12, 15),
        "beta": (15, 25),
        "gamma": (25, 40),
    }

    # Initialize dictionary for storing features
    features = {}
    freqs = epoch_spectrum.freqs
    # Loop through each channel to compute features
    for channel in epoch_spectrum.ch_names:
        psd = epoch_spectrum.get_data(picks=channel)
        # Compute total power for normalization
        total_power = psd.sum()
        if (
            total_power == 0
        ):  # if the channel is flat treat like uniform distribution to avoid 0 division
            psd = np.ones_like(psd)
            total_power = psd.sum()
        # 2. Normalize the PSD to a probability density function (area = 1 between [0.3-40Hz])
        normalized_psd = psd / total_power
        # 3. Log scale transformation and Normalize the log-scaled PSD
        low_non_zero_value = np.min(normalized_psd[np.nonzero(normalized_psd)])
        log_psd = np.log(
            normalized_psd * (np.exp(1) / low_non_zero_value)
        )  # Avoid negative values

        log_total_power = log_psd.sum()
        normalized_log_psd = log_psd / log_total_power

        # 8. Calculate entropy (non-log and log PSD)
        features[f"{channel}_root_total_power"] = np.sqrt(total_power)
        features[f"{channel}_spectral_entropy"] = entropy(normalized_psd[0, 0, :])

        # 3. Calculate power in specified frequency bands
        for band_name, (fmin, fmax) in band_dict.items():
            # Find frequency indices for the band
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)

            # Estimate band power
            band_power = normalized_psd[0, 0, idx_band].sum()
            log_band_power = normalized_log_psd[0, 0, idx_band].sum()
            # Save the log relative and relative band power feature
            features[f"{channel}_{band_name}_power"] = band_power
            features[f"{channel}_log_{band_name}_power"] = log_band_power

        # 7. Linear regression on the log-log PSD

        freqs_reshaped = np.log(freqs.reshape(-1, 1))  # Reshape for linear regression

        lin_reg_log = LinearRegression().fit(
            freqs_reshaped, len(freqs_reshaped) * normalized_log_psd[0, 0, :]
        )

        features[f"{channel}_log_spectral_slope"] = lin_reg_log.coef_[0]
        features[f"{channel}_log_spectral_intercept"] = lin_reg_log.intercept_

    return (
        features  # The returned structure is a dictionary where feature names are keys
    )


def extract_eog_features(eog_epoch_spectrum):
    # Initialize dictionary for storing features
    # refractory_period = 0.4  # dont expect more then 2 per second

    features = {}

    mapper = {
        "EOG1": "vertical_left",
        "EOG2": "horizontal_right",
        "EOGR": "r_l_diff",
    }

    for channel in eog_epoch_spectrum.ch_names:
        psd = eog_epoch_spectrum.get_data(picks=channel)
        # Compute total power of the chanel
        features[f"EOG_{mapper[channel]}"] = psd[0, 0, :].sum()

    return features


def extract_ecg_features(ecg_epoch_data):

    # Initialize dictionary for storing features
    refractory_period = 60 / 220  # minimal beat to beat interval in seconds bpm 220
    features = {}
    sample_frequency = ecg_epoch_data.info["sfreq"]

    for channel in ecg_epoch_data.info["ch_names"]:
        ecg = ecg_epoch_data.get_data()
        squared_signal = np.diff(ecg[0, 0, :]) ** 2  # emphasis peaks
        # identify peaks
        peaks, __annotations__ = signal.find_peaks(
            squared_signal,
            height=(np.mean(squared_signal), None),
            distance=int(refractory_period * sample_frequency),
        )

        # calculate rr intervals
        rr_intervals = np.diff(peaks) / sample_frequency
        norm_rr_intervals = rr_intervals / np.mean(rr_intervals)
        # calculate features
        features[f"{channel}_heart_rate"] = 60 / np.mean(rr_intervals)
        features[f"{channel}_SDNN"] = np.std(norm_rr_intervals)
        features[f"{channel}_rmssd"] = np.sqrt(np.mean(np.diff(norm_rr_intervals) ** 2))

    return features


def extract_emg_features(emg_epoch_spectrum):

    # Initialize dictionary for storing features
    features = {}  # (6-50)
    band_dict = {"power": (10, 50)}
    freqs = emg_epoch_spectrum.freqs

    for channel in emg_epoch_spectrum.ch_names:
        psd = emg_epoch_spectrum.get_data(picks=channel)
        # Compute total power in each band
        for band_name, (fmin, fmax) in band_dict.items():
            # Find frequency indices for the band
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            # Estimate band power
            features[f"{channel}_{band_name}"] = psd[0, 0, idx_band].sum()

    return (
        features  # The returned structure is a dictionary where feature names are keys
    )


# %%
if __name__ == "__main__":

    base_path = os.getcwd()
    txt_files = list(
        filter(lambda x: x.startswith("E"), glob.glob("**/*.txt", recursive=True))
    )

    channel_positions = pd.read_csv(
        "Co-registered average positions.pos",
        header=None,
        delimiter="\t",
        names=["electrode", "x", "y", "z"],
    )

    with open("analysis_code/parameters.yaml", "r") as file:
        PARAMETERS = yaml.safe_load(file)

    emg_channels = PARAMETERS["emg_channels"]
    eog_channels = PARAMETERS["eog_channels"]
    ecg_channels = PARAMETERS["ecg_channels"]

    channel_type_mapping = {}
    for ch in emg_channels:
        channel_type_mapping[ch] = "emg"

    for ch in ecg_channels:
        channel_type_mapping[ch] = "ecg"

    for ch in eog_channels:
        channel_type_mapping[ch] = "eog"

    for ch in channel_positions.electrode.values:
        channel_type_mapping[ch.upper()] = "eeg"

    pos_dict = {}
    for _, row in channel_positions.iterrows():
        pos_dict[row["electrode"].upper()] = [row["x"], row["y"], row["z"]]

    montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame="head")

    for name in reversed(txt_files):
        hypno = read_hypno_and_plot(Path(name))
        raw = mne.io.read_raw_edf(Path(name.replace(".txt", ".edf")))
        raw = raw.rename_channels(lambda x: x.replace("-Ref", "").upper())
        raw = raw.set_channel_types(channel_type_mapping)
        raw = raw.set_montage(montage)

        recording_start = raw.info["meas_date"]
        # pre-process
        epochs, means, stds, recording_time = preprocess_eeg(raw)
        # estimate the PSD per epoch
        spectrums = epochs.compute_psd(
            method="welch", fmin=0.5, fmax=40, n_fft=4096, remove_dc=False, picks="eeg"
        )

        emg_spectrums = epochs.compute_psd(
            method="welch", fmin=10, fmax=50, n_fft=2048, remove_dc=False, picks="emg"
        )

        eog_spectrums = epochs.compute_psd(
            method="welch", fmin=0.3, fmax=2, n_fft=2048, remove_dc=False, picks="eog"
        )

        # extract_features, save raw features
        # power band, log power bands, from non EEG features extract root total power of the time series
        features = {}
        for i in range(len(spectrums)):
            features[i] = extract_spectral_features(spectrums[i])

        for i in range(len(emg_spectrums)):
            features[i].update(extract_emg_features(emg_spectrums[i]))

        for i in range(len(eog_spectrums)):
            features[i].update(extract_eog_features(eog_spectrums[i]))

        for i in range(len(epochs)):
            features[i].update(extract_ecg_features(epochs[i].load_data().pick("ecg")))

        hypno["patient"] = name.split("\\")[-1].replace(".txt", "")
        hypno["time"] = hypno["time_from_onset"].apply(
            lambda x: recording_start + pd.Timedelta(seconds=x)
        )
        hypno.set_index("time", inplace=True)

        # add timeline label, sleep stage
        feature_matrix = pd.DataFrame(features).T.reset_index(drop=False)
        feature_matrix["time"] = feature_matrix["index"].apply(
            lambda x: recording_start + pd.Timedelta(seconds=x * 30)
        )
        feature_matrix = feature_matrix.set_index("time").drop(columns=["index"])
        pd.concat([hypno, feature_matrix], axis=1, join="inner").to_csv(
            name.replace(".txt", ".features.csv")
        )
        # clean and normalize features

        means = feature_matrix.mean()
        stds = feature_matrix.std()
        feature_matrix_norm = (feature_matrix - means) / stds
        feature_matrix_norm = feature_matrix_norm.clip(upper=4)
        # save standardization features
        pd.concat([hypno, feature_matrix_norm], axis=1, join="inner").to_csv(
            name.replace(".txt", ".scaled_features.csv")
        )


print("done")

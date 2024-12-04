import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import mne
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.getcwd()))
from preprocessing_utils import (
    preprocess_eeg,
    extract_spectral_features,
    extract_emg_features,
    extract_eog_features,
    extract_ecg_features,
    read_hypno_and_plot,
)


if __name__ == "__main__":

    # Load base path and files
    base_path = os.getcwd()
    txt_files = list(
        filter(lambda x: x.startswith("E"), glob.glob("**/*.txt", recursive=True))
    )

    # Load channel positions
    channel_positions = pd.read_csv(
        "Co-registered average positions.pos",
        header=None,
        delimiter="\t",
        names=["electrode", "x", "y", "z"],
    )

    # Load parameters from YAML file
    with open("analysis_code/parameters.yaml", "r") as file:
        PARAMETERS = yaml.safe_load(file)

    OUTPUT_PATH = Path(PARAMETERS["OUTPUT_DIR"]) / "features"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    window_length = PARAMETERS["window_length"]
    # Extract channel groups from parameters
    emg_channels = PARAMETERS["emg_channels"]
    eog_channels = PARAMETERS["eog_channels"]
    ecg_channels = PARAMETERS["ecg_channels"]

    # Mapping channels to their types
    channel_type_mapping = {ch: "emg" for ch in emg_channels}
    channel_type_mapping.update({ch: "ecg" for ch in ecg_channels})
    channel_type_mapping.update({ch: "eog" for ch in eog_channels})

    for ch in channel_positions.electrode.values:
        channel_type_mapping[ch.upper()] = "eeg"

    # Create montage using channel positions
    pos_dict = {
        row["electrode"].upper(): [row["x"], row["y"], row["z"]]
        for _, row in channel_positions.iterrows()
    }
    montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame="head")

    # Process each file
    for name in tqdm(txt_files):

        raw = mne.io.read_raw_edf(Path(name.replace(".txt", ".edf")))
        raw = raw.rename_channels(lambda x: x.replace("-Ref", "").upper())
        raw = raw.set_channel_types(channel_type_mapping)
        raw = raw.set_montage(montage)

        recording_start = np.datetime64(raw.info["meas_date"])

        # Preprocess EEG
        epochs, means, stds, recording_time = preprocess_eeg(raw, window_length)
        window_samples = int(epochs[0].get_data().shape[-1])
        optimal_window = int(raw.info["sfreq"] * 4)
        # Compute power spectral density (PSD) for each signal type
        spectrums = epochs.compute_psd(
            method="welch",
            fmin=0.5,
            fmax=40,
            n_fft=min(optimal_window, window_samples),
            remove_dc=False,
            picks="eeg",
        )
        emg_spectrums = epochs.compute_psd(
            method="welch",
            fmin=10,
            fmax=50,
            n_fft=min(optimal_window, window_samples),
            remove_dc=False,
            picks="emg",
        )
        eog_spectrums = epochs.compute_psd(
            method="welch",
            fmin=0.3,
            fmax=2,
            n_fft=min(optimal_window, window_samples),
            remove_dc=False,
            picks="eog",
        )

        # Extract features for each epoch
        features = {}
        print("extracting spectral...")
        for i in tqdm(range(len(spectrums))):
            features[i] = extract_spectral_features(
                spectrums[i], band_dict=PARAMETERS["bands"]
            )

        print("extracting emg...")
        for i in tqdm(range(len(emg_spectrums))):
            features[i].update(extract_emg_features(emg_spectrums[i]))

        print("extracting eog...")
        for i in tqdm(range(len(eog_spectrums))):
            features[i].update(extract_eog_features(eog_spectrums[i]))

        print("extracting ecg...")
        for i in tqdm(range(len(epochs))):
            features[i].update(extract_ecg_features(epochs[i].load_data().pick("ecg")))

        hypno = read_hypno_and_plot(Path(name))
        # Process hypnogram data and feature matrix
        hypno["patient"] = name.split("\\")[-1].replace(".txt", "")

        hypno["time"] = hypno["time_from_onset"].apply(
            lambda x: recording_start + np.timedelta64(x, "s")
        )
        hypno.set_index("time", inplace=True)

        # resample the hypnogram based on window length:
        if window_length != 30:
            hypno = hypno.reindex(
                hypno.index.union(
                    [hypno.index[-1] + np.timedelta64(30 - window_length, "s")]
                )
            ).ffill()
            hypno = hypno.resample(f"{window_length}s", origin="start").ffill()
            hypno["epoch_length"] = window_length
            hypno["time_from_onset"] = np.arange(
                hypno.loc[recording_start, "time_from_onset"],
                len(hypno) * window_length,
                window_length,
            )

        # Create feature matrix
        feature_matrix = pd.DataFrame(features).T.reset_index(drop=False)
        feature_matrix["time"] = feature_matrix["index"].apply(
            lambda x: recording_start + np.timedelta64(x * window_length, "s")
        )
        feature_matrix = feature_matrix.set_index("time").drop(columns=["index"])
        subject = name.split("\\")[0]
        # Save raw features
        pd.concat([hypno, feature_matrix], axis=1, join="inner").to_csv(
            OUTPUT_PATH / f"{subject}.csv"
        )

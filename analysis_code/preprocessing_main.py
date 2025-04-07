import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import mne
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed


sys.path.append(os.path.abspath(os.getcwd()))
from preprocessing_utils import (
    preprocess_eeg,
    extract_spectral_features,
    extract_emg_features,
    extract_eog_features,
    extract_ecg_features,
    read_hypno_and_plot,
)


def process_spectral_features(j, psd_batch, freqs, ch_names, band_dict):
    return j, extract_spectral_features(
        psd_batch[j - i, :, :],
        freqs=freqs,
        ch_names=ch_names,
        band_dict=band_dict,
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
    n_jobs = PARAMETERS["n_jobs"]
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    window_length = PARAMETERS["window_length"]
    step_size = PARAMETERS["step_size"]
    batch_size = PARAMETERS["batch_size"]
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

        # Preprocess EEG
        # Define paths for saving/loading preprocessed data
        preprocessed_path = Path(name).parent / "preprocessed"
        os.makedirs(preprocessed_path, exist_ok=True)

        # Run preprocessing and save results
        epochs, means, stds, recording_time, recording_start, events = preprocess_eeg(
            name,
            montage,
            channel_type_mapping,
            window_length,
            overlap=window_length - step_size,
        )

        window_samples = int(epochs[0].get_data().shape[-1])
        optimal_window = int(epochs.info["sfreq"] * 4)

        # Compute power spectral density (PSD) for each signal type
        # Extract features for each epoch
        features = {}

        # Process the epochs in batches
        print("extracting spectral batches...")
        for i in tqdm(np.arange(0, len(events), batch_size)):
            # Compute PSD for the current batch without averaging across epochs.
            psd = epochs[i : i + batch_size].compute_psd(
                method="welch",
                n_fft=min(optimal_window, window_samples),
                fmin=0.5,
                fmax=40,
                remove_dc=False,
                picks="eeg",
                average="mean",
                verbose=False,
            )

            freqs = psd.freqs
            ch_names = psd.ch_names
            psd = psd.get_data()

            batch_size_psd = psd.shape[0]

            # Process batch in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_spectral_features)(
                    j, psd, freqs, ch_names, PARAMETERS["bands"]
                )
                for j in range(i, i + batch_size_psd)
            )

            # Update features dictionary with results
            for j, feature_dict in results:
                features[j] = feature_dict

        emg_spectrums = epochs.compute_psd(
            method="welch",
            fmin=10,
            fmax=50,
            n_fft=min(optimal_window, window_samples),
            remove_dc=False,
            picks="emg",
        )

        print("extracting emg...")
        results = Parallel(n_jobs=-1)(
            delayed(extract_emg_features)(emg_spectrums[i])
            for i in tqdm(range(len(emg_spectrums)))
        )
        for i, feature_dict in enumerate(results):
            features[i].update(feature_dict)

        eog_spectrums = epochs.compute_psd(
            method="welch",
            fmin=0.3,
            fmax=2,
            n_fft=min(optimal_window, window_samples),
            remove_dc=False,
            picks="eog",
        )

        print("extracting eog...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(extract_eog_features)(eog_spectrums[i])
            for i in tqdm(range(len(eog_spectrums)))
        )
        for i, feature_dict in enumerate(results):
            features[i].update(feature_dict)

        del eog_spectrums

        print("extracting ecg...")
        print("extracting ecg...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(extract_ecg_features)(epochs[i].load_data().pick("ecg"))
            for i in tqdm(range(len(epochs)))
        )
        for i, feature_dict in enumerate(results):
            features[i].update(feature_dict)

        hypno = read_hypno_and_plot(Path(name))
        # Process hypnogram data and feature matrix
        hypno["patient"] = name.split("\\")[-1].replace(".txt", "")

        hypno["time"] = hypno["time_from_onset"].apply(
            lambda x: recording_start + np.timedelta64(x, "s")
        )
        hypno.set_index("time", inplace=True)

        # resample the hypnogram based on window length:
        if step_size != 30:
            hypno = hypno.reindex(
                hypno.index.union(
                    [hypno.index[-1] + np.timedelta64(30 - step_size, "s")]
                )
            ).ffill()
            hypno = hypno.resample(f"{step_size}s", origin="start").ffill()
            hypno["epoch_length"] = step_size
            hypno["time_from_onset"] = np.arange(
                hypno.loc[recording_start, "time_from_onset"],
                len(hypno) * step_size,
                step_size,
            )

        # Create feature matrix
        feature_matrix = pd.DataFrame(features).T.reset_index(drop=False)
        feature_matrix["time"] = feature_matrix["index"].apply(
            lambda x: recording_start + np.timedelta64(x * step_size, "s")
        )
        feature_matrix = feature_matrix.set_index("time").drop(columns=["index"])
        subject = name.split("\\")[0]
        # Save raw features
        pd.concat([hypno, feature_matrix], axis=1, join="inner").to_csv(
            OUTPUT_PATH / f"{subject}.csv"
        )

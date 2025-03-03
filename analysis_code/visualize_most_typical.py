import mne

import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import skew
from datetime import timedelta
import matplotlib as mpl


with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)


edf_dir = Path(PARAMETERS["DATA_DIR"])
data_dir = Path(PARAMETERS["DATA_DIR"])
pca_ica = "ica"

subject_state_files = list(
    data_dir.rglob("output_short/hmm/ica_5/subjects/*/shared_model/state_assignment_results.csv")
)


subject_state_files.sort()
subject_edf_files = list(edf_dir.rglob("*/*.edf"))
subject_edf_files.sort()
assert len(subject_state_files) == len(subject_edf_files)

psg_montage = {
    "F3-Ref": "F3",
    "FZ-Ref": "Fz",
    "F4-Ref": "F4",
    "C3-Ref": "C3",
    "CZ-Ref": "Cz",
    "C4-Ref": "C4",
    "P3-Ref": "P3",
    "PZ-Ref": "Pz",
    "P4-Ref": "P4",
    "O1-Ref": "O1",
    "OZ-Ref": "Oz",
    "O2-Ref": "O2",
    "ECG": "ECG",
    "EOG1": "EOG left",
    "EOG2": "EOG right",
    "EOGR": "EOG Relative",
    "ChEMG": "Chin EMG",
}

elect_type = {
    "F3-Ref": "eeg",
    "F4-Ref": "eeg",
    "C3-Ref": "eeg",
    "C4-Ref": "eeg",
    "P3-Ref": "eeg",
    "P4-Ref": "eeg",
    "O1-Ref": "eeg",
    "O2-Ref": "eeg",
    "FZ-Ref": "eeg",
    "PZ-Ref": "eeg",
    "OZ-Ref": "eeg",
    "CZ-Ref": "eeg",
    "EOG1": "eog",
    "EOG2": "eog",
    "EOGR": "eog",
    "ChEMG": "emg",
    "ECG": "ecg",
}

figure_dir = (
    Path(os.getcwd().replace("high_density_sleep", "")) / f"eeg-viewer-{pca_ica}" / "figures"
)
os.makedirs(figure_dir, exist_ok=True)

for states_path, eeg_path in zip(subject_state_files, subject_edf_files):
    subject = states_path.parts[-3]
    df = pd.read_csv(states_path, index_col="Unnamed: 0")
    feature_df = pd.read_csv(
        data_dir / f"output_short/features/{subject}.csv", parse_dates=["time"]
    )

    if pd.to_datetime(feature_df["time"].values[0]).hour < 17:
        time_correction = 5
    else:
        time_correction = 0

    feature_df = feature_df[feature_df["stage"] != "L"]
    feature_df["time"] = feature_df["time"].apply(
        lambda x: x + timedelta(hours=time_correction)
    )
    raw = mne.io.read_raw_edf(eeg_path, preload=True)
    raw.set_meas_date(raw.info["meas_date"] + timedelta(hours=time_correction))
    df["time_from_file_onset_seconds"] = (
        (feature_df["time"] - raw.info["meas_date"].replace(tzinfo=None))
        .apply(lambda x: int(x.total_seconds()))
        .values
    )

    most_probable_epochs = df.loc[
        df.groupby("hidden_states")[list(map(str, range(7)))].idxmax().max(axis=1)
    ]

    raw = mne.set_bipolar_reference(
        raw,
        anode=["RLEG+", "LLEG+", "ChEMG1", "ECG1"],
        cathode=["RLEG-", "LLEG-", "ChEMG2", "ECG2"],
        ch_name=["RLEG", "LLEG", "ChEMG", "ECG"],
        drop_refs=True,
    )
    raw = mne.set_bipolar_reference(
        raw, anode=["EOG1"], cathode=["EOG2"], ch_name=["EOGR"], drop_refs=False
    )

    raw.pick(list(psg_montage.keys()))
    raw.set_channel_types(elect_type)
    raw.rename_channels(psg_montage)
    raw.resample(250)
    raw.filter(l_freq=0.5, h_freq=40, picks="eeg")
    raw = raw.filter(l_freq=0.3, h_freq=6, picks="eog", n_jobs=1)
    raw = raw.filter(l_freq=20, h_freq=50, picks="emg", n_jobs=1)
    raw = raw.filter(l_freq=1, h_freq=40, picks="ecg", n_jobs=1)

    df["annot_names"] = [
        f"state {s}-{a}"
        for s, a in zip(df["hidden_states"], df["hidden_states_mapped"])
    ]

    my_annot = mne.Annotations(
        onset=df["time_from_file_onset_seconds"].values,  # in seconds
        duration=[4] * len(df),  # in seconds, too
        description=df["annot_names"],
        orig_time=raw.info["meas_date"],
    )

    raw.set_annotations(my_annot)

    events_from_annot, event_dict = mne.events_from_annotations(raw)

    event_colors = {
        desc: PARAMETERS[f"state_color_map_{pca_ica}"][desc] for desc in event_dict.keys()
    }
    custom_colors = list(event_colors.values())
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=custom_colors)

    data = raw.to_data_frame()

    eeg_s = 50e-6
    eog_s = 150  # data_std[["EOG left", "EOG right", "EOG Relative"]].median()
    emg_s = 50  # data_std["Chin EMG"] * 5
    ecg_s = 1e-3 * np.sign(skew(data["ECG"]))

    for i, row in most_probable_epochs.iterrows():
        state = row["hidden_states"]
        time = row["time_from_file_onset_seconds"]
        plot_start = int(time) - 12
        plot_end = int(time) + 18
        inx = np.logical_and(
            df["time_from_file_onset_seconds"] >= plot_start,
            df["time_from_file_onset_seconds"] < plot_end,
        )
        event_colors[-1] = "gray"
        fig = mne.viz.plot_raw(
            raw,
            duration=30,
            start=plot_start,
            scalings=dict(
                eeg=eeg_s,
                eog=eog_s,
                ecg=ecg_s,
                emg=emg_s,
            ),
            time_format="clock",
            # events=events_from_annot,         # <--- Include the events here
            # event_id=event_dict,             # <--- ...and the event_id mapping
            # event_color=event_colors,
            show=False,
        )

        fig.savefig(figure_dir / f"{eeg_path.name.replace('.edf', '')}_{state}.svg")
        plt.close()

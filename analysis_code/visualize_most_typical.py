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
from scipy.stats import skew
from datetime import timedelta
import matplotlib as mpl


with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)


edf_dir = Path(PARAMETERS["DATA_DIR"])
output_dir = Path(PARAMETERS["OUTPUT_DIR"])
epoch_duration = PARAMETERS["step_size"]

pca_ica = "ica"

folder = list(filter(lambda x: x.startswith(pca_ica), os.listdir(output_dir / "hmm")))[
    0
]

hmm_dir = output_dir / "hmm" / folder / "common_model"

state_assignment_results = pd.read_csv(
    hmm_dir / "state_assignment_results.csv", index_col=0
)


subject_edf_files = list(edf_dir.rglob("*/*.edf"))
subject_edf_files.sort()
assert state_assignment_results["patient"].nunique() == len(subject_edf_files)

psg_montage = [
    "F3",
    "F4",
    "C3",
    "C4",
    "O1",
    "O2",
    "ECG",
    "EOGR",
    "CHEMG",
]

figure_dir = output_dir / f"eeg-viewer-{pca_ica}" / "figures"
os.makedirs(figure_dir, exist_ok=True)

for eeg_path in subject_edf_files:
    subject = eeg_path.stem
    df = state_assignment_results.query(f"patient == '{subject}'")
    feature_df = pd.read_csv(
        output_dir / "features" / f"{subject}.csv", parse_dates=["time"]
    )

    if pd.to_datetime(feature_df["time"].values[0]).hour < 17:
        time_correction = 5
    else:
        time_correction = 0

    feature_df = feature_df[feature_df["stage"] != "L"]
    feature_df["time"] = feature_df["time"].apply(
        lambda x: x + timedelta(hours=time_correction)
    )

    raw = (
        mne.io.read_raw_fif(
            eeg_path.parent / "preprocessed" / f"{eeg_path.stem}_raw.fif", preload=True
        )
        .load_data()
        .resample(100)
    )

    raw.pick(psg_montage)

    raw.set_meas_date(raw.info["meas_date"] + timedelta(hours=time_correction))

    df["time_from_file_onset_seconds"] = (
        (feature_df["time"] - raw.info["meas_date"].replace(tzinfo=None))
        .apply(lambda x: int(x.total_seconds()))
        .values
    )

    n_states = df["hidden_states"].nunique()
    most_probable_epochs = df.loc[
        df.groupby("hidden_states")[list(map(str, range(n_states)))]
        .idxmax()
        .max(axis=1)
    ]

    df["annot_names"] = [
        f"state {s}-{a}"
        for s, a in zip(df["hidden_states"], df["hidden_states_mapped"])
    ]

    my_annot = mne.Annotations(
        onset=df["time_from_file_onset_seconds"].values,  # in seconds
        duration=[epoch_duration] * len(df),  # in seconds, too
        description=df["annot_names"],
        orig_time=raw.info["meas_date"],
    )

    raw.set_annotations(my_annot)

    events_from_annot, event_dict = mne.events_from_annotations(raw)

    state_color_map = PARAMETERS["stage_color_map"]
    states = event_dict.keys()

    event_colors = {}

    for s in state_color_map:
        base_color = state_color_map[s]
        related_states = list(filter(lambda x: x.endswith(s), states))
        alpha = 1
        for state in related_states:
            # Create a color palette with variations of the base color
            # by adjusting the lightness/saturation
            base_rgb = list(mpl.colors.to_rgb(base_color))
            # Create a color map from the base color to a slightly lighter version (rgba)
            state_color = base_rgb.copy()
            state_color.append(alpha)
            # Get the color for this state from the palette
            event_colors[state] = state_color
            alpha = alpha - 0.1

    custom_colors = [event_colors[state] for state in states]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=custom_colors)

    data = raw.to_data_frame() * 1e-6

    eeg_s = 50e-6
    eog_s = data["EOGR"].quantile(
        0.975
    )  #  150  # data_std[["EOG left", "EOG right", "EOG Relative"]].median()
    emg_s = data["CHEMG"].quantile(0.99)  # 50  # data_std["Chin EMG"] * 5
    ecg_s = data["ECG"].quantile(0.999)

    for i, row in most_probable_epochs.iterrows():
        state = row["hidden_states"]
        time = row["time_from_file_onset_seconds"]
        plot_start = int(time)
        plot_end = int(time) + 30
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

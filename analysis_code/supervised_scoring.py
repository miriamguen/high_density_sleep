# This script is used to score the supervised model
# in this script we will use the PCA AND ICA components
# 1. to train a linear svm model
# 2. do so in a cross-subject manner (leave one subject out cross validation)
# 3. score the model using agreement for each state, presenting the confusion matrix

import glob
from pathlib import Path
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from tqdm import tqdm
import yasa

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
)


os.getcwd()


def resample_yasa(hypno_df, subject_data):
    last_time = hypno_df["time"].iloc[-1]
    start_time = subject_data["time"].iloc[0]
    last_index = hypno_df.index.values[-1]
    hypno_df.loc[last_index + 1, ["time", "stage", "patient"]] = hypno_df[
        ["time", "stage", "patient"]
    ].iloc[-1]
    hypno_df.loc[last_index, "time"] = subject_data["time"].iloc[-1]
    hypno_df = (
        hypno_df.set_index("time")
        .resample(f"{window_step}s", origin=start_time)
        .ffill()
        .bfill()
        .reset_index()
    )
    keep = hypno_df.time.apply(lambda x: x in subject_data.time.values)
    hypno_df = hypno_df.loc[keep, :]
    return hypno_df


with open("analysis_code/parameters.yaml", "r") as f:
    PARAMETERS = yaml.safe_load(f)

output_dir = Path(PARAMETERS["OUTPUT_DIR"])
data_dir = Path(PARAMETERS["DATA_DIR"])
window_step = PARAMETERS["step_size"]

data = pd.read_parquet(output_dir / "decomposition" / "models" / "all_pc_data.parquet")
data["time"] = pd.to_datetime(data["time"])
subjects_metadata = pd.read_csv(
    data_dir / "Details information for healthy subjects.csv"
)

txt_files = list(
    filter(lambda x: x.startswith("E"), glob.glob("**/*.txt", recursive=True))
)

save_dir = output_dir / "supervised_scoring"
save_dir.mkdir(parents=True, exist_ok=True)
y = "stage"
stages = ["W", "R", "N1", "N2", "N3"]
pca_ica = "pca"

over_all_metrics = {}
over_all_metrics_yasa = {}
all_yasa_labels = []
max_comp = 63
min_comp = 1

if os.path.exists(save_dir / f"{pca_ica}_metrics_df.csv"):
    metrics_df = pd.read_csv(save_dir / f"{pca_ica}_metrics_df.csv")
    over_all_metrics_yasa_df = pd.read_csv(save_dir / f"yasa_metrics_df.csv")
else:
    for subject in tqdm(data["patient"].unique()):
        subject_data = data[data["patient"] == subject]
        train_data = data[data["patient"] != subject]
        eeg_file_path = data_dir / f"{subject}/{subject}/preprocessed/{subject}_raw.fif"
        save_name = data_dir / f"{subject}/{subject}/yasa_hypno.csv"
        gender_male = (
            subjects_metadata.loc[subjects_metadata["Subjects ID"] == subject, "Sex"]
            == "M"
        ).values[0]
        age = subjects_metadata.loc[
            subjects_metadata["Subjects ID"] == subject, "Age"
        ].values[0]

        if os.path.exists(save_name):
            # Add a last sample with similar timestamp as subject data by copying the last entry
            if window_step == 30:
                hypno_df = pd.read_csv(save_name, parse_dates=["time"])

            elif os.path.exists(
                save_name.parent / f"{save_name.stem}_resampled_{window_step}.csv"
            ):
                hypno_df = pd.read_csv(
                    save_name.parent / f"{save_name.stem}_resampled_{window_step}.csv",
                    parse_dates=["time"],
                )

            else:
                hypno_df = pd.read_csv(save_name, parse_dates=["time"])
                hypno_df = resample_yasa(hypno_df, subject_data)
                hypno_df.to_csv(
                    save_name.parent / f"{save_name.stem}_resampled_{window_step}.csv",
                    index=False,
                )

            all_yasa_labels.append(hypno_df)
        else:
            raw = mne.io.read_raw_fif(eeg_file_path, preload=True).load_data()
            raw = mne.set_bipolar_reference(
                raw,
                anode=["C4", "EOG1", "LLEG"],
                cathode=["TP9", "TP10", "RLEG"],
                ch_name=["C4-M1", "LOC-M2", "EMG1-EMG2"],
                drop_refs=True,
            )
            # Resample to 100 Hz if needed (YASA's default sampling rate)
            raw.resample(100)

            # Load the best channel for YASA sleep scoring
            hypno = yasa.SleepStaging(
                raw,
                eeg_name="C4-M1",
                eog_name="LOC-M2",
                emg_name="EMG1-EMG2",
                metadata=dict(age=age, male=gender_male),
            )

            # Get the predicted sleep stages
            hypno_pred = hypno.predict()

            # Create DataFrame with the predictions
            hypno_df = pd.DataFrame(
                {
                    "stage": hypno_pred,
                    "patient": subject,
                    "time": np.datetime64(raw.info["meas_date"])
                    + pd.TimedeltaIndex(
                        np.arange(0, len(hypno_pred)) * PARAMETERS["step_size"], "s"
                    ),
                }
            ).set_index("time")

            hypno_df = hypno_df.loc[
                subject_data.time, :
            ].reset_index()  # use only the labeled epochs

            if window_step != 30:
                hypno_df = resample_yasa(hypno_df, subject_data)
                save_name = (
                    save_name.parent / f"{save_name.stem}_resampled_{window_step}.csv"
                )

            hypno_df.to_csv(save_name, index=False)
            all_yasa_labels.append(hypno_df)

        over_all_metrics_yasa[subject] = {
            "kappa": cohen_kappa_score(subject_data["stage"], hypno_df["stage"]),
            "accuracy": accuracy_score(subject_data["stage"], hypno_df["stage"]),
        }

        over_all_metrics_yasa[subject].update(
            {
                f"precision_{stage}": score
                for stage, score in zip(
                    stages,
                    precision_score(
                        subject_data["stage"],
                        hypno_df["stage"],
                        labels=stages,
                        average=None,
                    ),
                )
            }
        )

        over_all_metrics_yasa[subject].update(
            {
                f"recall_{stage}": score
                for stage, score in zip(
                    stages,
                    recall_score(
                        subject_data["stage"],
                        hypno_df["stage"],
                        labels=stages,
                        average=None,
                    ),
                )
            }
        )

        over_all_metrics_yasa[subject].update(
            {
                f"f1_{stage}": score
                for stage, score in zip(
                    stages,
                    f1_score(
                        subject_data["stage"],
                        hypno_df["stage"],
                        labels=stages,
                        average=None,
                    ),
                )
            }
        )

        for i in tqdm(range(min_comp, max_comp)):
            train_on = data.columns[data.columns.str.startswith(pca_ica)][0:i]

            X_train = train_data[train_on]
            y_train = train_data[y]

            X_test = subject_data[train_on]
            y_test = subject_data[y]

            model = LinearSVC(class_weight="balanced")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            run_name = f"{i}_{subject}"

            result = {
                f"precision_{stage}": x
                for stage, x in zip(
                    stages, precision_score(y_test, y_pred, labels=stages, average=None)
                )
            }
            result.update(
                {
                    f"recall_{stage}": x
                    for stage, x in zip(
                        stages,
                        recall_score(y_test, y_pred, labels=stages, average=None),
                    )
                }
            )
            result.update(
                {
                    f"f1_{stage}": x
                    for stage, x in zip(
                        stages, f1_score(y_test, y_pred, labels=stages, average=None)
                    )
                }
            )
            result["kappa"] = cohen_kappa_score(y_test, y_pred)
            result["accuracy"] = accuracy_score(y_test, y_pred)
            result["components"] = i

            over_all_metrics[run_name] = result

    metrics_df = pd.DataFrame(over_all_metrics).T
    metrics_df.to_csv(save_dir / f"{pca_ica}_metrics_df.csv", index=False)
    over_all_metrics_yasa_df = pd.DataFrame(over_all_metrics_yasa).T
    over_all_metrics_yasa_df.to_csv(save_dir / f"yasa_metrics_df.csv", index=False)

# Set legend, axis label, and tick label font sizes
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

over_all_metrics_yasa_summary = over_all_metrics_yasa_df.describe()


fig, ax = plt.subplots(4, 1, figsize=(16, 16), sharex="col")


sns.lineplot(
    data=metrics_df,
    x="components",
    y="accuracy",
    ax=ax[0],
    errorbar=("ci", 95),
    label="Accuracy",
    color="gray",
)

ax[0].axhline(
    over_all_metrics_yasa_summary.loc["mean", "accuracy"],
    label="Accuracy YASA",
    color="gray",
    linestyle="--",
    lw=0.75,
)


sns.lineplot(
    data=metrics_df,
    x="components",
    y="kappa",
    ax=ax[0],
    errorbar=("ci", 95),
    label="Kappa",
    color="black",
)

ax[0].axhline(
    over_all_metrics_yasa_summary.loc["mean", "kappa"],
    label="Kappa YASA",
    color="black",
    linestyle="--",
    lw=0.75,
)

ax[0].set_ylabel("General Measures")
ax[0].set_ylim(0.3, 1)
ax[0].legend(bbox_to_anchor=(1.05, 1))

stage_colors = PARAMETERS["stage_color_map"]

# precision by stage
for stage in stages:
    sns.lineplot(
        data=metrics_df,
        x="components",
        y=f"precision_{stage}",
        ax=ax[1],
        errorbar=("ci", 95),
        label=stage,
        color=stage_colors[stage],
    )
    ax[1].axhline(
        over_all_metrics_yasa_summary.loc["mean", f"precision_{stage}"],
        label=f"{stage} YASA",
        color=stage_colors[stage],
        linestyle="--",
        lw=0.75,
    )

ax[1].set_ylabel("Precision")
ax[1].set_ylim(-0.02, 1)
# Add legend for sleep stages
ax[1].legend(bbox_to_anchor=(1.05, 1))
# recall by stage
for stage in stages:
    sns.lineplot(
        data=metrics_df,
        x="components",
        y=f"recall_{stage}",
        ax=ax[2],
        errorbar=("ci", 95),
        color=stage_colors[stage],
    )

    ax[2].axhline(
        over_all_metrics_yasa_summary.loc["mean", f"recall_{stage}"],
        label=f"{stage} YASA",
        color=stage_colors[stage],
        linestyle="--",
        lw=0.75,
    )

ax[2].set_ylabel("Recall")
ax[2].set_ylim(-0.02, 1)
ax[2].legend().remove()
# f1 by stage
for stage in stages:
    sns.lineplot(
        data=metrics_df,
        x="components",
        y=f"f1_{stage}",
        ax=ax[3],
        errorbar=("ci", 95),
        color=stage_colors[stage],
    )

    ax[3].axhline(
        over_all_metrics_yasa_summary.loc["mean", f"f1_{stage}"],
        label=f"{stage} YASA",
        color=stage_colors[stage],
        linestyle="--",
        lw=0.75,
    )

ax[3].legend().remove()
ax[3].set_xlabel("Number of Components")
ax[3].set_xticks(range(1, max_comp, 5))
ax[3].set_xticklabels(range(1, max_comp, 5))
ax[3].set_ylabel("F1")
ax[3].set_ylim(-0.02, 1)


fig.tight_layout()
fig.savefig(save_dir / "metrics_by_comp_num.svg")
plt.close()

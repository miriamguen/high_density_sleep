"""This script will

1. Load the standardized features of all patients and exclude epochs not labeled as one of the [W, N1-3, REM]
2. Decompose the signals using PCA and keeping 95% of the explained variance ratio
3. Estimate the EX of each PC and PCs accounting for more then 5% of the data variability will be further analyzed
    a. The top feature contribution maps will be drawn:
        - EEG features these will be represented on topographic maps
        - EOG, ECG, EMG a vertical bar plot will be created one for power and one for entropy
    b. the top component time series will be drawn together with the hypnogram for each patient
    c. The cross patient 3D low dimension representation plot will be create using the main PCS and colored by sleep stage
    d. The data will be clustered K-means, estimating optimal cluster number and
    f. The relation between sleep stages and patient ID will be estimated using confusion matrices and agreement with the sleep stages
    e. The top pc values will linearly modeled relative to each the sleep category to investigating the linear relation between each PC and the sleep stages
    f. model the pcs ex per patient to sleep quality measures of the patient - anova
    h. By the dominant stage in each cluster will be used to assign a state to each cluster, and decoding accuracy for each class will be estimated


"""

import os
from pathlib import Path
import glob
import yaml
from typing import Union
from joblib import dump
import itertools

import numpy as np
import pandas as pd

from picard import picard
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
)

from scipy.optimize import linear_sum_assignment
import scipy.linalg as linalg
import statsmodels.api as sm
import pingouin as pg


import mne
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import ticker
import plotly.graph_objects as go

mpl.use("svg")
plt.close("all")


def load_data_and_clean(
    path: Path,
    metadata_columns: list,
    parameters: dict,
) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw DataFrame and remove segments that do not include a sleep label.
    Standardize the data by calculating the mean and std for each feature by sleep stage,
    then calculate a global mean/std across stages and apply Z-scoring.
    Clip values to the range [-5, 5].

    Args:
        path (Path): The path to the raw data file.

    Returns:
        clean_data (pd.DataFrame): Cleaned and standardized feature DataFrame.
        means (pd.DataFrame): DataFrame containing mean values per sleep stage, and the overall used for normalization
        stds (pd.DataFrame): DataFrame containing std values per sleep stage, and the overall used for normalization
        dropped_segments: int
        clipped segments: int
    """
    # Load the data
    raw_data = pd.read_csv(path)
    feature_col = list(
        filter(lambda x: x not in metadata_columns, raw_data.columns.values)
    )
    # Exclude epochs not labeled as W, N1, N2, N3, R
    valid_stages = parameters["valid_sleep_stages"]
    indexes = raw_data["stage"].isin(valid_stages)
    dropped_segments = len(raw_data) - sum(indexes)
    raw_data = raw_data[indexes]

    # Calculate mean and std for each feature within each sleep stage
    grouped = raw_data[feature_col + ["stage"]].groupby("stage")
    means = grouped.apply(lambda x: np.mean(x, axis=0)).rename(
        lambda x: f"mean_{x}", axis=0
    )
    stds = grouped.apply(lambda x: np.std(x, axis=0)).rename(
        lambda x: f"stds_{x}", axis=0
    )

    # Calculate the global mean and std (averaged across stages)
    means.loc["global_mean", :] = means.mean()
    means["file"] = str(path)
    stds.loc["global_stds", :] = stds.mean()
    stds["file"] = str(path)
    # Z-score the data

    clean_data = (
        raw_data.loc[:, feature_col] - means.loc["global_mean", feature_col]
    ) / stds.loc["global_stds", feature_col]

    clipped = {
        "upper": (clean_data < -5).sum(axis=0).sort_values().sum() / clean_data.size,
        "lower": (clean_data > 5).sum(axis=0).sort_values().sum() / clean_data.size,
    }
    # Clip the data to limit extreme values to avoid a strong outlier drive for PCA

    clean_data = clean_data.clip(lower=-5, upper=5)
    clean_data[metadata_columns] = raw_data[metadata_columns]
    return clean_data, means, stds, dropped_segments, clipped


def rename_pc(auto_name):
    if "a" not in auto_name:
        return auto_name
    else:
        if type(auto_name) == str:
            pair = auto_name.upper().split("A")
        else:
            pair = auto_name.get_text().upper().split("A")
        num = int(pair[1]) + 1
        return f"{pair[0]} {num}"


def plot_hypno_with_components(
    patient_transformed_data,
    hypno_name,
    main_pc_names,
    explained_variance,
    explained_variance_patient,
    figure_save_path,
    parameters,
):
    n_time_series = len(main_pc_names) + 1
    fig, axes = plt.subplots(
        n_time_series, 1, figsize=(12, 3 + n_time_series), sharex="col"
    )

    y_ticks_labels = parameters["sleep_stge_plot_values"]

    labels = (
        patient_transformed_data[hypno_name].apply(lambda x: y_ticks_labels[x]).values
    )
    time_from_onset = patient_transformed_data["time_from_onset"].values
    time_from_onset = time_from_onset - time_from_onset[0]
    # Plot horizontal lines with vertical transitions (stair-like)
    for i in range(len(labels) - 2):
        color = "black"
        if labels[i] == 4:
            color = "red"
        elif labels[i] == 1:
            color = "navy"

        # Plot horizontal line
        axes[0].hlines(
            labels[i],
            time_from_onset[i],
            time_from_onset[i + 1],
            colors=color,
            linewidth=2,
        )

        # Plot vertical line to connect different stages
        if labels[i] != labels[i + 1]:
            axes[0].vlines(
                time_from_onset[i + 1],
                labels[i],
                labels[i + 1],
                colors="black",
                linewidth=2,
            )

        # Fill the background colors above and below the lines
        axes[0].fill_between(
            time_from_onset,
            labels + 0.05,
            5.0,
            color="lightblue",
            alpha=0.5,
            step="post",
        )

        # Set y-ticks and labels
        axes[0].set_yticks(list(y_ticks_labels.values()))
        axes[0].set_yticklabels(list(y_ticks_labels.keys()))

        # Set labels and title
        axes[0].set_xlabel("Time from Onset (hours)")
        axes[0].set_ylabel("Sleep Stage")

    for i, pc in enumerate(main_pc_names):
        axes[i + 1].plot(
            time_from_onset, patient_transformed_data[pc], color="lightgray", alpha=0.7
        )
        axes[i + 1].plot(
            time_from_onset,
            patient_transformed_data[pc].rolling(2).mean(),
            color="black",
        )
        axes[i + 1].set_ylabel(f"{rename_pc(pc)} (PDU)")
        axes[i + 1].set_title(
            f"explained variance ratio: all subjects-{explained_variance[i]:.2f}, subjects-{explained_variance_patient[pc]:.2f}"
        )

    # Set x-ticks to mark every hour
    xticks = np.arange(0, max(time_from_onset) + 1, 3600)
    xticklabels = [
        f"{int(x/3600)}h" for x in np.arange(0, max(time_from_onset) + 1, 3600)
    ]

    axes[n_time_series - 1].set_xticks(xticks)
    axes[n_time_series - 1].set_xticklabels(xticklabels)
    fig.tight_layout()
    os.makedirs(Path(figure_save_path) / "by_patient", exist_ok=True)
    fig.savefig(
        Path(figure_save_path)
        / "by_patient"
        / f"{patient_transformed_data['patient'].values[0]}_hypno_with_components.svg"
    )

    plt.close()


def decompose_all_patients(
    data,
    metadata_columns,
    patient_id_col,
    hypno_col,
    ex,
    figure_save_path: str = "decomposition",
    model_save_path: str = "models",
) -> Union[PCA, list, pd.DataFrame, pd.DataFrame, dict]:
    """
    Perform PCA on the standardized data, re center the sheared data for  identify components explaining at least 5% of variance.
    Plot hypnogram and top PC time series. Calculate explained variance per patient and plot distribution.

    Args:
        data (pd.DataFrame): Standardized patient data.
        metadata_columns (list): Columns that should not be included in the PCA (like metadata).
        patient_id_col (str): Column name for the patient ID.
        hypno_col (str): Column name for the hypnogram (sleep stage) column.
        ex (float): Explained variance threshold.
        figure_save_path (str): Directory to save the figures.
        model_save_path

    Returns:
        pca_model (PCA): The fitted PCA model.
        pc_names: (List): the pc names indicating flip
        transformed_data (pd.DataFrame): Data transformed by PCA.
        transformed_metadata (pd.DataFrame): Metadata associated with each patient.
        ex_by_patient (dict): Explained variance per patient and component.
    """

    """

    2. decompose all patient data, identify all components explaining at least 5% of variance (make sure pre-whitening is deactivated)
    3. draw the hypno time series, and below the top PC time series, and save figure in the feature folder or 'figure_save_path'
    4. calculate the ex per patient and for each of the top components and save to a dictionary
    5. plot the distribution for the patient ex for each component in a plot like the ones used in meta analysis (with the CI for each pc)
    """
    os.makedirs(figure_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    features = data.drop(columns=metadata_columns)
    means_overall = features.mean()
    centered_data = features - means_overall

    pca_model = PCA(random_state=42, whiten=False, n_components=ex)
    pca_model.set_output(transform="pandas")
    pca_model.fit(centered_data)
    transformed_data = pca_model.transform(centered_data)

    dump(pca_model, Path(model_save_path) / "pca_model.joblib")
    means_overall.to_csv(
        Path(model_save_path) / "feature_means_for_centering_joined.csv"
    )

    explained_variance = pca_model.explained_variance_ratio_
    pc_names = list(transformed_data.columns.values)
    pc_weights = pd.DataFrame(
        data=pca_model.components_,
        index=pc_names,
        columns=pca_model.feature_names_in_,
    ).T

    sleep_stge_plot_values = PARAMETERS["sleep_stge_plot_values"]
    hypno_values = data[hypno_col].apply(lambda x: sleep_stge_plot_values[x]).values

    # invert the time series and weights is inverted so if needed for graphical representation
    for i, pc in enumerate(pc_names):
        direct = np.sign(np.corrcoef(hypno_values, transformed_data[pc].values)[0, 1])
        if direct < 0:
            transformed_data[pc] = -1 * transformed_data[pc]
            transformed_data.rename({pc: f"-{pc}"}, axis=1, inplace=True)
            pc_weights[pc] = -1 * pc_weights[pc]
            pc_weights.rename({pc: f"-{pc}"}, axis=1, inplace=True)
            pc_names[i] = f"-{pc}"

    # use pcs with over 2% ex for ICA
    pc_names_ic = pc_names[: sum(explained_variance > 0.01)]
    ic_names = [x.replace("p", "i").replace("-", "") for x in pc_names_ic]
    total_ex_ic = sum(explained_variance[: sum(explained_variance > 0.01)])

    K, W, S = picard(
        transformed_data[pc_names_ic].values.T,
        fun="tanh",
        ortho=True,
        extended=True,
        centering=False,
        whiten=True,
        random_state=1,
        fastica_it=None,
        max_iter=1000,
    )
    print("finished ICA")
    W = np.dot(W, K)

    ic_var = np.var(S.T, axis=0)
    ic_ex = ic_var / ic_var.sum() * total_ex_ic

    ind = np.argsort(-ic_ex)  # sort the ICs by the EX variance they provide

    ic_ex = pd.Series(data=ic_ex[ind], index=ic_names, name="explained_variance")

    transformed_data[ic_names] = S[ind, :].T

    ic_weights = pd.DataFrame(data=W[ind, :], index=ic_names, columns=pc_names_ic).T

    ic_weights.to_csv(Path(model_save_path) / "pca_to_ica_model_wights.csv")

    for i, ic in enumerate(ic_names):
        direct = np.sign(np.corrcoef(hypno_values, transformed_data[ic].values)[0, 1])
        if direct < 0:
            transformed_data[ic] = -1 * transformed_data[ic]
            transformed_data.rename({ic: f"-{ic}"}, axis=1, inplace=True)
            ic_weights[ic] = -1 * ic_weights[ic]
            ic_weights.rename({ic: f"-{ic}"}, axis=1, inplace=True)
            ic_names[i] = f"-{ic}"

    transformed_data[metadata_columns] = data[metadata_columns]
    centered_data[metadata_columns] = data[metadata_columns]
    print("saving data...")
    transformed_data.to_parquet(Path(model_save_path) / "all_pc_data.parquet")
    centered_data.to_parquet(Path(model_save_path) / "centered_data.parquet")
    print("done saving data...")
    # 3. Plot hypnogram and top PC time series for each patient
    patients = data[patient_id_col].unique()
    ex_by_patient = {}
    os.makedirs(Path(figure_save_path) / "pca", exist_ok=True)
    os.makedirs(Path(figure_save_path) / "ica", exist_ok=True)
    for patient in patients:
        patient_transformed_data = transformed_data[
            transformed_data[patient_id_col] == patient
        ]
        patient_data = centered_data.drop(columns=metadata_columns)[
            transformed_data[patient_id_col] == patient
        ]

        ex_by_patient[patient] = (
            np.var(patient_transformed_data[pc_names + ic_names], axis=0)
            / np.var(patient_data, axis=0).sum()
        ).to_dict()

        print(f"plotting {patient} hypno with components")
        plot_hypno_with_components(
            patient_transformed_data,
            hypno_col,
            pc_names_ic,
            explained_variance,
            ex_by_patient[patient],
            figure_save_path="decomposition\\pca",
        )

        plot_hypno_with_components(
            patient_transformed_data,
            hypno_col,
            ic_names,
            ic_ex.values,
            ex_by_patient[patient],
            figure_save_path="decomposition\\ica",
        )

    ex_data = pd.DataFrame(ex_by_patient).T
    # plot the distribution for the patient ex for each component in a plot like the ones used in meta analysis (with the CI for each pc)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(
        data=ex_data[pc_names_ic],
        inner="point",
        palette=sns.color_palette("Blues_r", len(pc_names_ic)),
    )
    ax.set_title("PC explained variance distribution", fontsize=16)
    ax.set_xlabel("Component", fontsize=14)
    ax.set_ylabel("Explained Variance", fontsize=14)
    ax.set_xticklabels([rename_pc(label._text) for label in ax.get_xticklabels()])
    plt.tight_layout()
    fig.savefig(Path(figure_save_path) / "pca" / "explained_variance_distribution.svg")
    plt.close()

    # plot the distribution for the patient ex for each component in a plot like the ones used in meta analysis (with the CI for each pc)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(
        data=ex_data[ic_names],
        inner="point",
        palette=sns.color_palette("Blues_r", len(ic_names)),
    )
    ax.set_title("Component explained variance distribution", fontsize=16)
    ax.set_xlabel("Component", fontsize=14)
    ax.set_ylabel("Explained Variance", fontsize=14)
    xticklabels = [rename_pc(label._text) for label in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels)
    plt.tight_layout()
    fig.savefig(Path(figure_save_path) / "ica" / "explained_variance_distribution.svg")
    plt.close()

    return (
        centered_data,
        transformed_data,
        ex_data,
        pca_model,
        pc_names,
        pc_names_ic,
        pc_weights,
        ic_names,
        ic_weights,
        ic_ex,
    )


def plot_interactive_low_dimension_map(
    transformed_data_df: pd.DataFrame,
    component_names: list,
    stage_col: str = "stage",
    figure_save_path: str = "decomposition",
    parameters: dict = None,
):
    """
    Plots an interactive 3D scatter plot of the first 3 principal components, colored by the labeled sleep stage.

    Args:
        transformed_data_df (pd.DataFrame): DataFrame containing the first 3 PC values and the sleep stage label.
        component_names(list): List containing the names of the PC axes to plot.
        stage_col (str): Name of the column indicating the labeled sleep stage.
        figure_save_path (str): Path to save the interactive figure.
    """
    stage_color_map = parameters["stage_color_map"]

    x, y, z = component_names
    fig = go.Figure()
    # Plot each stage with corresponding color
    for stage in stage_color_map.keys():
        stage_data = transformed_data_df[transformed_data_df[stage_col] == stage]

        # Main scatter plot with smaller markers
        fig.add_trace(
            go.Scatter3d(
                x=stage_data[x],
                y=stage_data[y],
                z=stage_data[z],
                mode="markers",
                marker=dict(
                    size=2,  #
                    color=stage_color_map[stage],
                    opacity=0.5,
                ),
                name=stage,
                showlegend=False,  # Hide this trace in the legend
            )
        )
        # Separate trace for the legend with larger markers
        # Dummy trace for the legend with larger markers
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],  # No actual points, just for the legend
                mode="markers",
                marker=dict(
                    size=12,  # Increase dot size for the legend
                    color=stage_color_map[stage],
                ),
                name=stage,
                showlegend=True,  # Show this trace in the legend
            )
        )

    # Update layout for 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x} (PDU)",
            yaxis_title=f"{y} (PDU)",
            zaxis_title=f"{z} (PDU)",
            aspectmode="cube",
        ),
        title="3D Scatter Plot of First 3 Principal Components by Sleep Stage",
        legend_title="Sleep Stage",
    )

    # Save the interactive plot as an HTML file
    fig.write_html(
        Path(figure_save_path) / f"{x}_{y}_{z}_3d_dimension_sleep_dynamics.html"
    )
    fig.show()


def plot_low_dimension_map(
    transformed_data_df: pd.DataFrame,
    component_names: list,
    stage_col: str,
    figure_save_path: str,
    parameters: dict,
):
    """
    Plots a 3D scatter plot of the first 3 components (by ex), colored by the labeled sleep stage.

    Args:
        transformed_data_df (pd.DataFrame): DataFrame containing the first 3 PC values and the sleep stage label.
        transformed_data: the names of the pc axes to plot
        stage_col (str): Name of the column indicating the labeled sleep stage.
        figure_save_path: figure_save_path
    """
    transformed_data_df = transformed_data_df.rename(
        columns=lambda x: rename_pc(x) if "ca" in x else x
    )
    stage_color_map = parameters["stage_color_map"]
    component_names = [rename_pc(x) for x in component_names]
    ordered_trip = list(itertools.combinations(np.arange(0, len(component_names)), 3))

    for trip in ordered_trip:
        # Create a 3D scatter plot
        plot_interactive_low_dimension_map(
            transformed_data_df=transformed_data_df,
            component_names=[component_names[i] for i in trip],
            stage_col=stage_col,
            figure_save_path=figure_save_path,
            parameters=parameters,
        )

    ordered_pairs = list(itertools.permutations(np.arange(0, len(component_names)), 2))

    # plot the 2d maps:
    for pair in ordered_pairs:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Plot each stage with corresponding color
        for stage in stage_color_map.keys():
            stage_data = transformed_data_df[transformed_data_df[stage_col] == stage]
            ax.scatter(
                stage_data[component_names[pair[0]]],
                stage_data[component_names[pair[1]]],
                c=stage_color_map[stage],
                label=stage,
                s=2,  # Adjust size of the markers
                alpha=0.5,  # Transparency for better visibility
            )

        # Add labels and a legend
        ax.set_xlabel(f"{component_names[pair[0]]} (PDU)")
        ax.set_ylabel(f"{component_names[pair[1]]} (PDU)")
        ax.set_title(
            f"2D Scatter Plot of {component_names[pair[0]]}-{component_names[pair[1]]} by Sleep Stage"
        )
        ax.legend(title="Sleep Stage", loc="upper left")

        fig.savefig(
            Path(figure_save_path)
            / f"{component_names[pair[0]]}_{component_names[pair[1]]}_2d_dimension_sleep_dynamics.svg"
        )
        plt.close()


def plot_component_weight_map(
    weights: pd.DataFrame,
    component_name: str,
    eeg_name_map: dict,
    non_eeg_name_map: dict,
    channel_positions: pd.DataFrame,  # [ x, y, z]
    title: str,
    save_path: Path,
    figure_columns: int = 8,
) -> plt.figure:
    """The feature maps"""
    channel_positions = channel_positions.reset_index()
    channel_positions["electrode"] = channel_positions["electrode"].str.upper()

    pos_dict = {}
    for _, row in channel_positions.iterrows():
        pos_dict[row["electrode"]] = [row["x"], row["y"], row["z"]]

    montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame="head")
    info = mne.create_info(
        ch_names=channel_positions["electrode"].to_list(), sfreq=1, ch_types="eeg"
    )
    info.set_montage(montage, match_case=False)

    non_eeg_features = list(
        filter(
            lambda x: x.split("_")[0] not in channel_positions["electrode"].values,
            weights.index.values,
        )
    )

    weights["feature"] = list(map(lambda x: "_".join(x.split("_")[1:]), weights.index))
    weights["electrode"] = list(map(lambda x: x.split("_")[0], weights.index))
    # separate the EEG from the non EEG weights

    non_eeg_weights = weights.loc[non_eeg_features, :]
    eeg_weights = weights.drop(index=non_eeg_features)

    total_subplots = 1 + len(eeg_name_map.keys()) + len(non_eeg_name_map.keys())
    rows = int(np.ceil(total_subplots / figure_columns))

    fig, ax = plt.subplots(rows, figure_columns, figsize=(3 * figure_columns, 3 * rows))
    gs = ax[rows - 1, figure_columns - 1].get_gridspec()

    for ax_ in ax[rows - 1, figure_columns - 1 :]:
        ax_.remove()
    colorbar_ax = fig.add_subplot(gs[rows - 1, figure_columns - 1 :])
    max_val = np.max(
        np.abs(weights[component_name])
    )  # To define a symmetric color range

    # plot the EEG maps
    for i, feature in enumerate(eeg_name_map.keys()):
        x, y = np.divmod(i, figure_columns)

        feature_weights = eeg_weights[eeg_weights.feature == feature]

        feature_weights.set_index("electrode", inplace=True)

        ax[x, y].set_title(eeg_name_map[feature], fontsize=16)

        mne.viz.plot_topomap(
            data=feature_weights[component_name].values,
            pos=info,  # feature_weights[["x", "y"]].values,
            axes=ax[x, y],
            ch_type="eeg",
            vlim=[-max_val, max_val],
            show=False,
            extrapolate="local",  #'box'
            sphere="auto",
            contours=4,
            # outlines=None
        )

    # plot the non EEG
    start_ind = len(eeg_name_map.keys())
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

    max_mod = max(non_eeg_weights[component_name].abs())
    for i, modality_group in enumerate(non_eeg_name_map.keys()):
        x, y = np.divmod(start_ind + i, figure_columns)
        mapper = non_eeg_name_map[modality_group]
        features_to_plot = list(mapper.keys())
        feature_weights = (
            non_eeg_weights.loc[features_to_plot, :]
            .rename(index=lambda x: mapper[x])
            .reset_index()
        )

        sns.barplot(
            feature_weights,
            y=component_name,
            x="index",
            hue=component_name,
            orient="v",
            ax=ax[x, y],
            palette=sns.color_palette("RdBu_r", as_cmap=True),
            hue_norm=norm,
            edgecolor="black",
            linewidth=0.5,
            legend=False,
        )
        sns.despine(bottom=True, left=True)
        ax[x, y].set_title(f"{modality_group}", fontsize=16)
        ax[x, y].tick_params(axis="x", rotation=90)
        ax[x, y].set_xlabel("")
        ax[x, y].set_ylabel("")
        ax[x, y].set_ylim([max(-2 * max_mod, -max_val), min(2 * max_mod, max_val)])
        ax[x, y].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    # %%

    norm = mpl.colors.Normalize(vmin=-max_val, vmax=max_val)

    colorbar = fig.colorbar(
        mappable=mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
        ax=colorbar_ax,
        use_gridspec=False,
        orientation="vertical",
        aspect=8,
        fraction=0.5,
        pad=0.5,
    )

    colorbar.ax.tick_params(labelsize=14, labelrotation=0, left=True)
    ax[x, y].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    colorbar.ax.set_xlabel("Weight scale (PDU)", fontsize=16)
    colorbar_ax.remove()

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path / f"{title.split(', ')[0]}_feature_map.svg")
    plt.close(fig)


def model_pc_vs_sleep_stage(
    transformed_data: pd.DataFrame,
    transformed_metadata: pd.DataFrame,
    patient_col: str = "patient",
    feature_cols: list = ["PC0"],
    label_col: str = "sleep_stage",
    save_path: str = "statistical_modeling",
):
    """
    Dependent: the sleep stage as nominal variable
    independent: the PC values of the top pcs
    random effect: patient

    # prepare data and export to datafreme the modeling will be performed in R
    """
    df = pd.concat(
        [
            transformed_metadata[label_col].astype("category"),
            transformed_metadata[patient_col].astype("category"),
            transformed_data[feature_cols],
        ],
        axis=1,
        join="inner",
    )

    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{save_path}\pc_to_sleep_stage.csv")


def model_patient_ex_and_sleep_quality(
    ex_by_patient: pd.DataFrame,
    sleep_quality_by_patient: pd.DataFrame,
    pcs: list = [],
    sleep_quality_measures: list = [],
):

    results_all = {}
    # Dictionary to store evaluation metrics for each model
    evaluation_metrics = {}
    X = ex_by_patient[pcs]
    X = sm.add_constant(X)
    for sqm in sleep_quality_measures:
        y = sleep_quality_by_patient[sqm]
        model = sm.OLS(y, X)
        results = model.fit()
        results_all[sqm] = results
        # Extract metrics
        evaluation_metrics[sqm] = {}
        evaluation_metrics[sqm]["R_squared"] = results.rsquared
        evaluation_metrics[sqm]["adj_R_squared"] = results.rsquared_adj
        evaluation_metrics[sqm]["residuals"] = results.resid
        coeff = {f"coeff_{key}": value for key, value in results.params.to_dict()}
        evaluation_metrics[sqm].update(coeff)
        p_values = {f"p_value_{key}": value for key, value in results.pvalues.to_dict()}
        evaluation_metrics[sqm].update(p_values)

    evaluation_metrics_df = pd.DataFrame(evaluation_metrics).T
    evaluation_metrics_df.reset_index(inplace=True)
    evaluation_metrics_df.rename(
        columns={"index": "sleep_quality_measure"}, inplace=True
    )

    ## Todo: plot significant results if exists
    # Plot R-squared and Adjusted R-squared for each sleep quality measure
    # Plot Coefficients for Each Feature
    # Plot P-values for Each Feature
    # Plot Residuals


# Function to align cluster labels with true labels
def map_cluster_labels(sleep_labels, cluster_labels):
    # Create the confusion matrix
    cm = pd.crosstab(
        pd.Series(sleep_labels, name="Sleep stage"),
        pd.Series(cluster_labels, name="Cluster"),
    )

    # Use the Hungarian algorithm to find the best mapping
    # Negative because we want max alignment
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create a mapping from cluster labels to true labels
    mapping = {
        cluster_label: true_label for cluster_label, true_label in zip(col_ind, row_ind)
    }

    # Map the cluster labels to the best matching true labels
    mapped_labels = np.vectorize(mapping.get)(cluster_labels)

    return mapped_labels


def cluster_analysis(transformed_data_, feature_names, result_path):
    """
    Performs KMeans clustering with various cluster sizes and computes metrics.

    Parameters:
    - transformed_data: DataFrame containing the data and labels.
    - feature_names: List of column names used for clustering.
    - result_path: a path to the result folder

    Returns:
    - Saves confusion matrices, metrics, and plots.
    - transformed_data: with the cluster labels included
    """
    transformed_data = transformed_data_.copy(deep=True)
    # Dictionary to store clustering metrics
    cluster_metrics = {
        "Silhouette Score": {},
        "Davies-Bouldin Index": {},
        "Adjusted Rand Index": {},
        "Normalized Mutual Information": {},
        "Adjusted Rand Index 5": {},
        "Normalized Mutual Information 5": {},
        "Adjusted Rand Index patient": {},
        "Normalized Mutual Information patient": {},
    }

    # Stage mappings for different K-values
    mapping_sets = {
        5: {"W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4},
        4: {"W": 0, "R": 1, "N1": 1, "N2": 2, "N3": 3},
        3: {"W": 0, "R": 1, "N1": 1, "N2": 2, "N3": 2},
        2: {"W": 0, "R": 1, "N1": 1, "N2": 1, "N3": 1},
    }

    # Reverse mappings for label display in confusion matrices
    invers_mapping_sets = {
        5: {0: "Wake", 1: "REM", 2: "N1", 3: "N2", 4: "N3"},
        4: {0: "Wake", 1: "transition", 2: "N2", 3: "N3"},
        3: {0: "Wake", 1: "transition", 2: "N2-3"},
        2: {0: "Wake", 1: "Sleep"},
    }

    # Create output directory for figures
    os.makedirs(result_path, exist_ok=True)

    for k in [2, 3, 4, 5]:

        label_mapping = mapping_sets[k]
        invers_map = invers_mapping_sets[k]
        label_col = f"hypno_num_{k}"
        transformed_data[label_col] = transformed_data["stage"].apply(
            lambda x: label_mapping[x]
        )

        kmeans = KMeans(n_clusters=k, random_state=42)

        labels = kmeans.fit_predict(transformed_data[feature_names])
        transformed_data[f"cluster_labels_{k}"] = map_cluster_labels(
            transformed_data[label_col].values, labels
        )
        # Compute Silhouette Score to measure cluster quality
        silhouette_avg = silhouette_score(
            transformed_data[feature_names], transformed_data[f"cluster_labels_{k}"]
        )
        cluster_metrics["Silhouette Score"][k] = silhouette_avg

        cluster_metrics["Davies-Bouldin Index"][k] = davies_bouldin_score(
            transformed_data[feature_names], transformed_data[f"cluster_labels_{k}"]
        )
        cluster_metrics["Adjusted Rand Index"][k] = adjusted_rand_score(
            transformed_data[label_col], transformed_data[f"cluster_labels_{k}"]
        )

        cluster_metrics["Adjusted Rand Index 5"][k] = adjusted_rand_score(
            transformed_data["stage"], transformed_data[f"cluster_labels_{k}"]
        )

        cluster_metrics["Adjusted Rand Index patient"][k] = adjusted_rand_score(
            transformed_data["stage"], transformed_data[f"cluster_labels_{k}"]
        )

        cluster_metrics["Normalized Mutual Information"][k] = (
            normalized_mutual_info_score(
                transformed_data[label_col], transformed_data[f"cluster_labels_{k}"]
            )
        )

        cluster_metrics["Normalized Mutual Information 5"][k] = (
            normalized_mutual_info_score(
                transformed_data["stage"], transformed_data[f"cluster_labels_{k}"]
            )
        )

        cluster_metrics["Normalized Mutual Information patient"][k] = (
            normalized_mutual_info_score(
                transformed_data["patient"], transformed_data[f"cluster_labels_{k}"]
            )
        )

        # Generate the confusion matrix between true labels and cluster labels
        confusion_matrix = pd.crosstab(
            transformed_data[label_col].values,
            transformed_data[f"cluster_labels_{k}"].values,
            rownames=["Sleep stage"],
            colnames=["Cluster"],
            dropna=True,
        ).rename(index=invers_map, columns=lambda x: f"Cluster-{x}")

        title = f"K = {k} confusion matrix"
        # Plot the confusion matrix
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
        plt.title(title)
        fig.savefig(f"{result_path}/{title}.svg")
        plt.close()
        confusion_matrix.to_csv(f"{result_path}/{title}.csv")

        confusion_matrix = (
            pd.crosstab(
                transformed_data["stage"].values,
                transformed_data[f"cluster_labels_{k}"].values,
                rownames=["Sleep stage"],
                colnames=["Cluster"],
                dropna=True,
            )
            .rename(columns=lambda x: f"Cluster-{x}")
            .loc[["W", "R", "N1", "N2", "N3"],]
        )

        title = f"K = {k} confusion matrix all"
        # Plot the confusion matrix
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
        plt.title(title)
        fig.savefig(f"{result_path}/{title}_all.svg")
        plt.close()
        confusion_matrix.to_csv(f"{result_path}/{title}.csv")

        confusion_matrix = pd.crosstab(
            transformed_data["patient"].values,
            transformed_data[f"cluster_labels_{k}"].values,
            rownames=["Sleep stage"],
            colnames=["Cluster"],
            dropna=True,
        ).rename(columns=lambda x: f"Cluster-{x}")

        title = f"K = {k} cluster by patient"
        # Plot the confusion matrix
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
        plt.title(title)
        fig.savefig(f"{result_path}/{title}_all.svg")
        plt.close()
        confusion_matrix.to_csv(f"{result_path}/{title}.csv")

    # Display the cluster quality metrics sorted by K
    pd.DataFrame(cluster_metrics).to_csv(f"{result_path}/cluster_metrics.csv")
    transformed_data.to_csv(f"{result_path}/transformed_data_with_cluster_labels.csv")


def bland_altman_plot(data1, data2, measure, save_path):
    """
    Creates and saves a Bland-Altman plot comparing two sets of measurements.
    Parameters:
        data1 (array-like): Measurements from the first method.
        data2 (array-like): Measurements from the second method.
        measure: str: the name of the measure compered
        save_path (str, optional): File path to save the plot. If None, the plot is not saved.
    """
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    y_span = np.max([np.std(mean) * 1.96, np.max(np.abs(diff))])
    y_span = y_span + y_span / 20
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color="gray", linestyle="--", label=f"Mean Difference ({md:.2f})")
    plt.axhline(
        md + 1.96 * sd,
        color="red",
        linestyle="--",
        label=f"+1.96 SD ({md + 1.96*sd:.2f})",
    )
    plt.axhline(
        md - 1.96 * sd,
        color="red",
        linestyle="--",
        label=f"-1.96 SD ({md - 1.96*sd:.2f})",
    )
    plt.xlabel("Mean of clusters and sleep labels")
    plt.ylabel("Difference between clusters and sleep labels")
    plt.ylim([-y_span, y_span])
    plt.title(f"Bland-Altman Plot {measure}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def calculate_icc(data1, data2):
    """
    Calculates the Intraclass Correlation Coefficient (ICC) between two sets of measurements.
    Parameters:
        data1 (pd.Series): Measurements from the first method.
        data2 (pd.Series): Measurements from the second method.
    Returns:
        icc_value (float): The ICC value.
        icc_details (pd.DataFrame): Detailed ICC results.
    """
    df_icc = pd.DataFrame(
        {
            "Subject": data1.index.tolist() + data2.index.tolist(),
            "Rater": ["Cluster"] * len(data1) + ["Sleep label"] * len(data2),
            "Value": np.concatenate([data1.values, data2.values]),
        }
    )
    icc_results = pg.intraclass_corr(
        data=df_icc, targets="Subject", raters="Rater", ratings="Value"
    )
    # Extract ICC2 (Two-way random effects, absolute agreement)
    icc_value = icc_results.loc[icc_results["Type"] == "ICC2", "ICC"].values[0]
    return icc_value, icc_results


# %%


if __name__ == "__main__":
    with open("analysis_code/parameters.yaml", "r") as file:
        PARAMETERS = yaml.safe_load(file)

    base_path = os.getcwd()
    # get a list of all feature matrices relative path
    feature_files = filter(
        lambda x: x.startswith("E"), glob.glob("**/*.features.csv", recursive=True)
    )

    metadata_columns = [
        "time",
        "stage",
        "time_from_onset",
        "epoch_length",
        "num",
        "patient",
    ]

    # %% Load data
    data_list = []
    means_list = []
    stds_list = []
    quality = {}
    for file in feature_files:
        clean_data, means, stds, dropped_segments, clipped = load_data_and_clean(
            Path(file), metadata_columns=metadata_columns, parameters=PARAMETERS
        )
        data_list.append(clean_data)
        means_list.append(means)
        stds_list.append(stds)
        clipped["dropped_segments"] = dropped_segments
        quality[file.split("\\")[-1].replace(".txt", "")] = clipped

    clean_data_all = pd.concat(data_list).reset_index(drop=True)

    (
        centered_data,
        transformed_data,
        ex_data,
        pca_model,
        pc_names,
        pc_names_ic,
        pc_weights,
        ic_names,
        ic_weights,
        ic_ex,
    ) = decompose_all_patients(
        data=clean_data_all,
        metadata_columns=metadata_columns,
        patient_id_col="patient",
        hypno_col="stage",  # the numeric hypnogram column
        ex=0.90,
        figure_save_path="decomposition",
    )

    # for visual representation flip the PC by their correlation

    # %% plot the pc maps per patient
    clean_non_eeg_channels = PARAMETERS["clean_non_eeg_channels"]

    emg_channels = PARAMETERS["emg_channels"]
    eog_channels = PARAMETERS["eog_channels"]
    ecg_channels = PARAMETERS["ecg_channels"]

    # plot the pc maps of PC accounting for at least 2% of the data variance

    # %% plot the cross subject 3 dimensional representation

    plot_low_dimension_map(
        transformed_data_df=transformed_data,
        component_names=pc_names_ic,
        stage_col="stage",
        figure_save_path="decomposition/pca",
    )

    plot_low_dimension_map(
        transformed_data_df=transformed_data,
        component_names=ic_names,
        stage_col="stage",
        figure_save_path="decomposition/ica",
    )
    # %% plot the pc activation maps
    # load the 2 d channel positions
    # channel_positions = pd.read_csv("pos_2_d.csv", index_col=0)

    channel_positions = pd.read_csv(
        "Co-registered average positions.pos",
        header=None,
        delimiter="\t",
        names=["electrode", "y", "x", "z"],
    )

    channel_positions.set_index("electrode", inplace=True)

    for i, pc_name in enumerate(pc_names_ic):
        plot_component_weight_map(
            weights=pc_weights[[pc_name]],
            component_name=pc_name,
            eeg_name_map=PARAMETERS["eeg_name_map"],
            non_eeg_name_map=PARAMETERS["non_eeg_name_map"],
            channel_positions=channel_positions,
            title=f"{rename_pc(pc_name)}, ex: {pca_model.explained_variance_ratio_[i]:.2f}",
            save_path=Path("decomposition/pca"),
        )

    feature_to_ic_weights = pd.DataFrame(
        data=linalg.blas.dgemm(
            alpha=1.0, a=pc_weights[pc_names_ic], b=ic_weights
        ),  # np.dot(pc_weights[strong_pcs], ic_weights),
        index=pc_weights.index,
        columns=ic_weights.columns,
    )

    for i, ic_name in enumerate(ic_names):
        plot_component_weight_map(
            weights=feature_to_ic_weights[[ic_name]],
            component_name=ic_name,
            eeg_name_map=PARAMETERS["eeg_name_map"],
            non_eeg_name_map=PARAMETERS["non_eeg_name_map"],
            channel_positions=channel_positions,
            title=f"{rename_pc(ic_name)}, ex: {ic_ex[i]:.2f}",
            save_path=Path("decomposition/ica"),
        )

    # %% cluster the data and compare to sleep labels (should be the same for ICA and PCA)

    cluster_analysis(transformed_data, pc_names_ic, "clustering/pca")
    cluster_analysis(transformed_data, ic_names, "clustering/ica")

    # %% Extract the sleep measures from the clusters
    def find_first_consecutive_non_W_index(labels, n=4, last_first="first"):
        count = 0  # To keep track of consecutive non-'W' values
        for i, value in enumerate(labels):
            if value != "W":
                count += 1  # Increment if non-'W' is found
                if count == n:
                    if last_first == "first":  # Check if we have n consecutive non-'W'
                        return i - n + 1
                    else:
                        return i  # Return the starting index
            else:
                count = 0  # Reset the counter if 'W' is encountered
        return None  # If no such sequence is found

    def find_rem_latancy(vec, n=4, rem_label="REM_N1"):
        count = 0
        # this assumes the patient is not narcoleptic and that REM will occure after passing through N3
        start_ind = vec[vec == "N3"].index[0]
        rem_n1 = vec[start_ind:]
        rem_n1 = rem_n1[rem_n1 == rem_label]
        rem_n1_gaps = np.diff(rem_n1.index.values)

        count = 1
        last_ind = rem_n1.index[0]
        max_count = [last_ind, 1]
        for i, gap in enumerate(rem_n1_gaps):
            if gap == 1:
                count += 1
                if count > max_count[1]:
                    max_count = [rem_n1.index[i + 1], count]
                if count == n:
                    return rem_n1.index[i + 1] - n + 1
            else:
                count = 1

        return (
            max_count[0] - max_count[1]
        )  # if a rem segment of the requiered size is not founs set the longest as the rem onset

    def get_sleep_masures_from_clusters_patient(patient_data, cluster_col, cluster_map):

        patient_data = patient_data.reset_index(drop=True)
        patient_data[cluster_col] = patient_data[cluster_col].apply(
            lambda x: cluster_map[x]
        )

        onset_index = find_first_consecutive_non_W_index(
            patient_data[cluster_col], n=10, last_first="first"
        )
        offset_index = len(patient_data) - find_first_consecutive_non_W_index(
            reversed(patient_data[cluster_col]), n=4, last_first="first"
        )

        sleep_range = range(onset_index, offset_index)

        cluster_measures = {}
        cluster_measures["TRT"] = len(patient_data) / 2
        cluster_measures["TST"] = (patient_data[cluster_col] != "W").sum() / 2
        cluster_measures["SE"] = 100 * cluster_measures["TST"] / cluster_measures["TRT"]
        cluster_measures["SOL"] = onset_index / 2
        cluster_measures["Awakenning"] = offset_index / 2
        cluster_measures["REML"] = (
            find_rem_latancy(patient_data[cluster_col], n=4) / 2
            - cluster_measures["SOL"]
        )
        cluster_measures["WASO"] = (
            np.sum(patient_data.loc[onset_index:, cluster_col] == "W") / 2
        )

        cluster_measures["N1"] = (
            np.sum(patient_data.iloc[onset_index:, :][cluster_col] == "REM_N1") / 4
        )
        cluster_measures["N2"] = (
            np.sum(patient_data.iloc[onset_index:, :][cluster_col] == "N2") / 2
        )
        cluster_measures["N3"] = (
            np.sum(patient_data.iloc[onset_index:, :][cluster_col] == "N3") / 2
        )
        cluster_measures["REM"] = (
            np.sum(patient_data.iloc[onset_index:, :][cluster_col] == "REM_N1") / 4
        )

        sleep_measures = {}
        onset_index = find_first_consecutive_non_W_index(
            patient_data["stage"], n=1, last_first="first"
        )
        offset_index = (
            len(patient_data)
            - find_first_consecutive_non_W_index(
                reversed(patient_data["stage"]), n=2, last_first="first"
            )
            + 1
        )

        sleep_measures["TRT"] = len(patient_data) / 2
        sleep_measures["TST"] = (patient_data["stage"] != "W").sum() / 2
        sleep_measures["SE"] = 100 * sleep_measures["TST"] / sleep_measures["TRT"]
        sleep_measures["SOL"] = onset_index / 2
        sleep_measures["Awakenning"] = offset_index / 2
        sleep_measures["REML"] = (
            patient_data[patient_data["stage"] == "R"].index.values[0] / 2
        ) - sleep_measures["SOL"]
        sleep_measures["WASO"] = (
            np.sum(patient_data.iloc[onset_index:, :]["stage"] == "W") / 2
        )
        sleep_measures["N1"] = (
            np.sum(patient_data.iloc[onset_index:, :]["stage"] == "N1") / 2
        )
        sleep_measures["N2"] = (
            np.sum(patient_data.iloc[onset_index:, :]["stage"] == "N2") / 2
        )
        sleep_measures["N3"] = (
            np.sum(patient_data.iloc[onset_index:, :]["stage"] == "N3") / 2
        )
        sleep_measures["REM"] = (
            np.sum(patient_data.iloc[onset_index:, :]["stage"] == "R") / 2
        )

        return cluster_measures, sleep_measures

    def compare_sleep_and_cluster_measures(sleep_measures_all, cluster_measures_all):
        icc2_result_summary = {}
        # Iterate over the columns (measures) and perform paired tests
        for col in sleep_measures_all.columns:
            if (col != "Patient") and (
                col != "TRT"
            ):  # Skip if there is a Patient ID column
                # Paired t-test
                gold_standard_results = sleep_measures_all[col].sort_index()
                new_test_results = cluster_measures_all[col].sort_index()
                assert all(
                    gold_standard_results.index == new_test_results.index
                ), "Indices of the Series do not match."
                # If the p-value indicates a non-normal distribution, use Wilcoxon test
                # Scatter plot to compare the two tests
                plt.figure(figsize=(8, 6))
                plt.scatter(gold_standard_results, new_test_results, alpha=0.7)
                max_val = max(gold_standard_results.max(), new_test_results.max())
                min_val = min(gold_standard_results.min(), new_test_results.min())
                margin = max_val / 20
                plt.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    lw=2,
                )
                plt.xlabel(f"{col} assesd with sleep labels")
                plt.ylabel(f"{col} assesd with clusters")
                plt.xlim((min_val - margin, max_val + margin))
                plt.ylim((min_val - margin, max_val + margin))
                plt.title(f"Comparind {col} estimation (Manual Vs. Unsupervised)")
                plt.grid(True)
                os.makedirs("sleep_measures/figures", exist_ok=True)
                plt.savefig(f"sleep_measures/figures/{col}_scatter_plot.svg")
                plt.close()

                # Calculate ICC
                icc_value, icc_details = calculate_icc(
                    new_test_results, gold_standard_results
                )
                icc2_result_summary[col] = (
                    icc_details.set_index("Type").loc["ICC2", :].to_dict()
                )
                print("Intraclass Correlation Coefficient (ICC):")
                print(f"ICC Value (Type ICC2): {icc_value:.3f}")
                print("\nDetailed ICC Results:")
                print(icc_details)

                # Save ICC results
                icc_details.to_csv(f"sleep_measures/{col}_icc_results.csv", index=False)

                # Bland-Altman Plot
                bland_altman_plot(
                    new_test_results.values,
                    gold_standard_results.values,
                    save_path=f"sleep_measures/figures/{col}_bland_altman_plot.svg",
                    measure=col,
                )
        pd.DataFrame(icc2_result_summary).to_csv(
            "sleep_measures/icc2_result_summary.csv"
        )

    data = pd.read_csv(
        "clustering/pca/transformed_data_with_cluster_labels.csv",
        usecols=["patient", "time_from_onset", "stage", "cluster_labels_5"],
    )

    cluster_map = {0: "W", 1: "REM_N1", 2: "N2", 3: "N2", 4: "N3"}
    cluster_col = "cluster_labels_5"
    sleep_measures_all = {}
    cluster_measures_all = {}

    for patient in list(set(data["patient"].values)):
        cluster_measures_all[patient], sleep_measures_all[patient] = (
            get_sleep_masures_from_clusters_patient(
                data[data["patient"] == patient], cluster_col, cluster_map
            )
        )

    cluster_measures_all = pd.DataFrame(cluster_measures_all).T.sort_index()
    sleep_measures_all = pd.DataFrame(sleep_measures_all).T.sort_index()

    os.makedirs("sleep_measures", exist_ok=True)
    cluster_measures_all.to_csv("sleep_measures/cluster_sleep_measures.csv")
    sleep_measures_all.to_csv("sleep_measures/label_sleep_measures.csv")

    compare_sleep_and_cluster_measures(sleep_measures_all, cluster_measures_all)

    # %% model the top pc values relative to each the sleep category to investigating the linear relation between each PC and the sleep stages
    model_pc_vs_sleep_stage(
        transformed_data=transformed_data,
        patient_col="patient",
        feature_cols=pc_names_ic,
        label_col="stage",
        save_path="statistical_modeling",
    )

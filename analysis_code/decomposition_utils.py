import os
from pathlib import Path
from joblib import dump

from typing import List, Dict, Tuple, Union
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "svg"
mpl.use("svg")
plt.close("all")

from sklearn.decomposition import PCA
from picard import picard
import mne


def load_data_and_clean(
    path: Path,
    metadata_columns: List[str],
    parameters: Dict[str, Union[List[str], str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    """
    Load the raw data from a CSV file, clean the data by excluding segments without valid sleep labels,
    and standardize the data by Z-scoring (mean and standard deviation) across sleep stages.
    Values are clipped to the range [-5, 5] to avoid the influence of extreme outliers.

    Args:
    -----
    path : Path
        The path to the raw features CSV file.
    metadata_columns : List[str]
        List of metadata column names that are excluded from Z-scoring and clipping.
    parameters : Dict[str, Union[List[str], str]]
        Dictionary of analysis parameters, including valid sleep stages used for data cleaning.

    Returns:
    --------
    clean_data : pd.DataFrame
        The cleaned and standardized feature DataFrame with Z-scored values.
    means : pd.DataFrame
        DataFrame containing mean values for each feature by sleep stage, including global means.
    stds : pd.DataFrame
        DataFrame containing standard deviation values for each feature by sleep stage, including global standard deviations.
    dropped_segments : int
        Number of segments that were dropped due to invalid or missing sleep stage labels.
    clipped : pd.DataFrame
        Dictionary containing the percentage of data points clipped at the upper and lower bounds.
    """
    # Load the raw data from CSV
    raw_data = pd.read_csv(path)

    # Filter out feature columns (excluding metadata columns)
    feature_col = list(
        filter(lambda x: x not in metadata_columns, raw_data.columns.values)
    )

    # Exclude epochs that don't have valid sleep stage labels (e.g., W, N1, N2, N3, R)
    valid_stages = parameters["valid_sleep_stages"]
    indexes = raw_data["stage"].isin(valid_stages)
    dropped_segments = len(raw_data) - sum(indexes)
    raw_data = raw_data[indexes]

    # Calculate mean and standard deviation for each feature within each sleep stage
    grouped = raw_data[feature_col + ["stage"]].groupby("stage")
    means = grouped.apply(lambda x: np.mean(x, axis=0)).rename(
        lambda x: f"mean_{x}", axis=0
    )
    stds = grouped.apply(lambda x: np.std(x, axis=0)).rename(
        lambda x: f"stds_{x}", axis=0
    )

    # Calculate the global mean and standard deviation (averaged across all sleep stages)
    means.loc["global_mean", :] = means.mean()
    means.loc["simple_mean", :] = np.mean(raw_data[feature_col], axis=0)

    means["file"] = str(path)
    stds.loc["global_stds", :] = stds.mean()
    stds.loc["simple_stds", :] = np.std(raw_data[feature_col], axis=0)
    stds["file"] = str(path)

    # Z-score the data using global mean and std
    # clean_data = (
    #     raw_data.loc[:, feature_col] - means.loc["global_mean", feature_col]
    # ) / stds.loc["global_stds", feature_col]

    # Z-score the data using the overall mean and std
    clean_data = (
        raw_data.loc[:, feature_col] - means.loc["simple_mean", feature_col]
    ) / stds.loc["simple_stds", feature_col]

    # Calculate the percentage of clipped values at the upper and lower bounds
    clipped = {
        "upper": (clean_data > 5).sum() / len(clean_data),
        "lower": (clean_data < -5).sum() / len(clean_data),
    }

    clipped = pd.DataFrame(clipped).T
    clipped["file"] = str(path)
    # Clip the data to the range [-5, 5] to limit extreme values
    missing = clean_data.isna().sum()
    clean_data = clean_data.apply(pd.to_numeric)
    clean_data = clean_data.interpolate("linear", axis=0)
    clean_data = clean_data.clip(lower=-5, upper=5)

    # Add metadata columns back to the cleaned data
    clean_data[metadata_columns] = raw_data[metadata_columns]

    return clean_data, means, stds, dropped_segments, clipped, missing


def rename_pc(auto_name: Union[str, object]) -> str:
    """
    Renames components based on the presence of the letter "a" in the string.
    If "a" is found, it splits the string at "A", increments the numeric portion, and returns a new name.
    Otherwise, it returns the original name.

    Args:
    -----
    auto_name : Union[str, object]
        The automatic name of the component, either as a string or an object with a `get_text()` method.

    Returns:
    --------
    str:
        The renamed component with an incremented numeric suffix if "a" is present; otherwise, returns the original name.
    """
    # Check if the name contains "a"
    if "a" not in auto_name:
        return auto_name
    else:
        # If input is a string, split the name at "A"
        if isinstance(auto_name, str):
            pair = auto_name.upper().split("A")
        else:
            # For non-string objects, call `get_text()` method to obtain the name
            pair = auto_name.get_text().upper().split("A")

        # Increment the numeric part of the name and return the new name
        num = int(pair[1]) + 1
        return f"{pair[0]} {num}"


def plot_hypno_with_components(
    patient_transformed_data: pd.DataFrame,
    hypno_name: str,
    main_pc_names: List[str],
    explained_variance: List[float],
    explained_variance_patient: Dict[str, float],
    figure_save_path: Path,
    parameters: Dict[str, Dict[str, int]],
):
    """
    Plots the hypnogram along with the primary components of PCA-transformed data for a specific patient,
    overlaying sleep stages with components and visualizing the explained variance of each principal component.

    Args:
    -----
    patient_transformed_data : pd.DataFrame
        The transformed data for a specific patient, including sleep stages and principal components.
    hypno_name : str
        The name of the column containing the hypnogram (sleep stages).
    main_pc_names : List[str]
        List of the main principal component names to be plotted.
    explained_variance : List[float]
        The variance explained by each principal component across all subjects.
    explained_variance_patient : Dict[str, float]
        The variance explained by each principal component for the specific patient.
    figure_save_path : Path
        The path where the figure will be saved.
    parameters : Dict[str, Dict[str, int]]
        Parameters, including sleep stage values for plotting.

    Returns:
    --------
    None
    """
    n_time_series = (
        len(main_pc_names) + 1
    )  # Number of time series (hypnogram + components)
    fig, axes = plt.subplots(
        n_time_series, 1, figsize=(12, int(1.5 * (1 + n_time_series))), sharex="col"
    )

    # Define y-ticks for sleep stage plotting based on parameters
    y_ticks_labels = parameters["sleep_stage_plot_values"]

    # Retrieve sleep stage labels and time from onset
    labels = (
        patient_transformed_data[hypno_name].apply(lambda x: y_ticks_labels[x]).values
    )
    time_from_onset = patient_transformed_data["time_from_onset"].values
    time_from_onset = time_from_onset - time_from_onset[0]  # Normalize to start at 0

    # Plot the hypnogram with vertical transitions
    stage_color_map = parameters["stage_color_map"]
    labels_colors = (
        patient_transformed_data[hypno_name].apply(lambda x: stage_color_map[x]).values
    )
    for i in range(len(labels) - 2):
        color = labels_colors[i]
        # color = "black"
        # if labels[i] == 4:  # REM stage
        #     color = "red"
        # elif labels[i] == 1:  # N1 stage
        #     color = "navy"

        # Horizontal line for sleep stages
        axes[0].hlines(
            labels[i],
            time_from_onset[i],
            time_from_onset[i + 1],
            colors=color,
            linewidth=1.5,
        )

        # Vertical line for transitions between sleep stages
        if labels[i] != labels[i + 1]:
            axes[0].vlines(
                time_from_onset[i + 1],
                labels[i],
                labels[i + 1],
                colors="gray",
                linewidth=1.5,
                alpha=0.5,
            )

    # Fill the background color (light blue) for the upper part of the plot
    axes[0].fill_between(
        time_from_onset, labels + 0.05, 5.0, color="lightblue", alpha=0.5, step="post"
    )

    # Set y-ticks and labels for sleep stages
    axes[0].set_yticks(list(y_ticks_labels.values()))
    axes[0].set_yticklabels(list(y_ticks_labels.keys()))
    axes[0].set_xlabel("Time from Onset (hours)")
    axes[0].set_ylabel("Sleep Stage")

    # Plot each principal component time series
    for i, pc in enumerate(main_pc_names):
        y_range = np.max(np.abs(patient_transformed_data[pc]))
        axes[i + 1].axhline(
            y=0, color="gray", linestyle="--", label="0", alpha=0.5, linewidth=0.5
        )
        axes[i + 1].plot(
            time_from_onset,
            patient_transformed_data[pc],
            color="gray",
            alpha=0.5,
            linewidth=1.5,
        )

        axes[i + 1].scatter(
            time_from_onset,
            patient_transformed_data[pc],
            c=labels_colors,
            alpha=0.7,
            s=1,
        )

        axes[i + 1].set_ylabel(f"{rename_pc(pc)} (PDU)")
        axes[i + 1].set_ylim([-y_range, y_range])

        # if pc.startswith("p"):
        axes[i + 1].set_title(
            f"Explained variance ratio: all subjects-{explained_variance[i]:.2f}, subject-{explained_variance_patient[pc]:.2f}"
        )

    # Set x-ticks to mark every hour
    xticks = np.arange(0, max(time_from_onset) + 1, 3600)
    xticklabels = [
        f"{int(x/3600)}h" for x in np.arange(0, max(time_from_onset) + 1, 3600)
    ]
    axes[n_time_series - 1].set_xticks(xticks)
    axes[n_time_series - 1].set_xticklabels(xticklabels)

    # Adjust layout and save the figure
    fig.tight_layout()
    os.makedirs(Path(figure_save_path) / "by_patient", exist_ok=True)
    fig.savefig(
        Path(figure_save_path)
        / "by_patient"
        / f"{patient_transformed_data['patient'].values[0]}_hypno_with_components.svg"
    )

    plt.close()


def decompose_all_patients(
    data: pd.DataFrame,
    metadata_columns: List[str],
    patient_id_col: str,
    hypno_col: str,
    ex: float,
    figure_save_path: Path,
    model_save_path: Path,
    parameters: dict,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    PCA,
    List[str],
    List[str],
    pd.DataFrame,
    List[str],
    pd.DataFrame,
    pd.Series,
]:
    """
    Perform PCA and ICA on standardized patient data, plot the hypnogram with components, and calculate
    the explained variance per patient. This function generates and saves the PCA and ICA decomposition
    models, visualizes the time series for the components, and calculates explained variance distributions.

    Args:
    -----
    data : pd.DataFrame
        The standardized feature data for all patients.
    metadata_columns : List[str]
        List of metadata column names that should not be included in the PCA.
    patient_id_col : str
        Column name for the patient ID.
    hypno_col : str
        Column name for the hypnogram (sleep stage) column.
    ex : float
        Explained variance threshold for PCA components.
    figure_save_path : Path
        Directory to save the figures (default is "decomposition").
    model_save_path : Path
        Directory to save the PCA/ICA models (default is "models").
    parameters : Dict the analysis parameter dictionary

    Returns:
    --------
    centered_data : pd.DataFrame
        The centered feature data.
    transformed_data : pd.DataFrame
        The PCA-transformed data.
    ex_data : pd.DataFrame
        DataFrame containing explained variance per patient for each component.
    pca_model : PCA
        The fitted PCA model.
    pc_names : List[str]
        Names of the principal components.
    pc_names_ic : List[str]
        Names of the independent components.
    pc_weights : pd.DataFrame
        Weights of the PCA components.
    ic_names : List[str]
        Names of the independent components.
    ic_weights : pd.DataFrame
        Weights of the ICA components.
    ic_ex : pd.Series
        Explained variance by each independent component.
    """

    # Create directories for saving figures and models
    os.makedirs(figure_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Drop metadata columns to get feature data and center it
    features = data.drop(columns=metadata_columns)
    # remove feature rows with over
    means_overall = features.mean()
    stds_overall = features.std()
    centered_data = (features - means_overall) / stds_overall

    # Perform PCA on the centered data
    pca_model = PCA(random_state=42, whiten=False, n_components=ex)
    pca_model.set_output(transform="pandas")
    pca_model.fit(centered_data)
    transformed_data = pca_model.transform(centered_data)

    # Save the PCA model and feature means
    dump(pca_model, Path(model_save_path) / "pca_model.joblib")
    means_overall.to_csv(Path(model_save_path) / "feature_means_for_centering.csv")
    stds_overall.to_csv(Path(model_save_path) / "feature_stds_for_centering.csv")

    explained_variance = pca_model.explained_variance_ratio_
    pc_names = list(transformed_data.columns.values)

    # PCA component weights
    pc_weights = pd.DataFrame(
        data=pca_model.components_,
        index=pc_names,
        columns=pca_model.feature_names_in_,
    ).T
    pc_weights.to_csv(model_save_path / "pca_weights.csv")
    # Map sleep stages to y-tick labels for plotting
    sleep_stge_plot_values = parameters["sleep_stage_plot_values"]
    hypno_values = data[hypno_col].apply(lambda x: sleep_stge_plot_values[x]).values

    # flip the component signs + wights for consistency with hypnogram for more intuitive visualization
    pc_corr = []
    for pc in pc_names:
        direct = np.corrcoef(hypno_values, transformed_data[pc].values)[0, 1]
        if direct < 0:
            transformed_data[pc] = -1 * transformed_data[pc]
            pc_weights[pc] = -1 * pc_weights[pc]
        pc_corr.append(direct)

    # Identify principal components explaining more than 1% of variance for ICA
    pc_names_ic = pc_names[: sum(explained_variance > parameters["min_ex"])]
    ic_names = [x.replace("p", "i").replace("-", "") for x in pc_names_ic]
    total_ex_ic = sum(
        explained_variance[: sum(explained_variance > parameters["min_ex"])]
    )

    # Perform ICA using Picard algorithm on the significant PCs
    K, W, S = picard(
        transformed_data[pc_names_ic].values.T,
        fun="tanh",
        ortho=False,
        lambda_min=0.001,
        extended=True,
        centering=False,
        whiten=True,
        random_state=1,
        tol=1e-14,
        fastica_it=None,
        max_iter=10000,
    )

    print("finished ICA")
    # W = np.dot(W, K)
    # Calculate explained variance for ICA components
    S = np.dot(W, transformed_data[pc_names_ic].values.T)
    ic_var = np.var(S.T, axis=0)
    features_var = np.var(features, axis=0)
    ic_ex = ic_var / features_var.sum()
    ic_corr = [np.corrcoef(hypno_values, S[ic, :])[0, 1] for ic in range(len(ic_names))]

    ind = np.argsort(-ic_ex)  # Sort ICs by overall explained variance
    ic_corr = [ic_corr[i] for i in ind]
    ic_ex = pd.DataFrame(
        data=ic_ex[ind], index=ic_names, columns=["explained_variance"]
    )
    ic_ex["hypno_corr"] = ic_corr
    transformed_data[ic_names] = S[ind, :].T

    # Save ICA component weights
    ic_weights = pd.DataFrame(data=W[ind, :], index=ic_names, columns=pc_names_ic).T
    ic_weights.to_csv(model_save_path / "pca_to_ica_weights.csv")
    
    # Adjust ICA component signs for consistency with hypnogram
    for i, ic in enumerate(ic_names):
        direct = ic_corr[i]
        if direct < 0:
            transformed_data[ic] = -1 * transformed_data[ic]
            ic_weights[ic] = -1 * ic_weights[ic]
            # transformed_data.rename({ic: f"-{ic}"}, axis=1, inplace=True)
            # ic_weights.rename({ic: f"-{ic}"}, axis=1, inplace=True)

    # Add metadata back to the transformed data
    transformed_data[metadata_columns] = data[metadata_columns]
    centered_data[metadata_columns] = data[metadata_columns]

    # Save transformed and centered data
    transformed_data.to_parquet(model_save_path / "all_pc_data.parquet")
    centered_data.to_parquet(model_save_path / "centered_data.parquet")

    # Plot hypnogram and component time series for each patient
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

        # Plot PCA and ICA components with hypnogram
        plot_hypno_with_components(
            patient_transformed_data,
            hypno_col,
            pc_names_ic,
            explained_variance,
            ex_by_patient[patient],
            figure_save_path=Path(figure_save_path) / "pca",
            parameters=parameters,
        )

        plot_hypno_with_components(
            patient_transformed_data,
            hypno_col,
            ic_names,
            ic_ex["explained_variance"].values,
            ex_by_patient[patient],
            figure_save_path=Path(figure_save_path) / "ica",
            parameters=parameters,
        )

    # Generate explained variance distribution plots for PCA and ICA components
    ex_data = pd.DataFrame(ex_by_patient).T

    ex_data.to_csv(model_save_path / "ex_by_patient.csv")

    ex_data_overall = pd.DataFrame(
        {"pc_ex": explained_variance, "pc_names": pc_names, "hypno_corr": pc_corr},
        index=range(1, len(pc_names) + 1),
    )

    ex_data_overall.to_csv(model_save_path / "pc_ex_overall.csv")

    ic_ex.to_csv(model_save_path / "ic_ex_overall.csv")

    # Plot PCA explained variance distribution
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(
        data=ex_data[pc_names_ic],
        inner="point",
        palette=sns.color_palette("Blues_r", len(pc_names_ic)),
    )
    ax.set_title("PCA Explained Variance Distribution", fontsize=16)
    ax.set_xlabel("Component", fontsize=14)
    ax.set_ylabel("Explained Variance", fontsize=14)
    ax.set_xticklabels([rename_pc(label.get_text()) for label in ax.get_xticklabels()])
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(Path(figure_save_path) / "pca" / "explained_variance_distribution.svg")
    plt.close()

    # Plot ICA explained variance distribution
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(
        data=ex_data[ic_names],
        inner="point",
        palette=sns.color_palette("Blues_r", len(ic_names)),
    )
    ax.set_title("ICA Explained Variance Distribution", fontsize=16)
    ax.set_xlabel("Component", fontsize=14)
    ax.set_ylabel("Explained Variance", fontsize=14)
    ax.set_xticklabels([rename_pc(label.get_text()) for label in ax.get_xticklabels()])
    ax.set_ylim(0, 1)
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
    component_names: List[str],
    stage_col: str = "stage",
    figure_save_path: str = "decomposition",
    parameters: Dict[str, Dict[str, str]] = None,
) -> None:
    """
    Plots an interactive 3D scatter plot of  3  components, colored by the labeled sleep stage.

    Args:
    -----
    transformed_data_df : pd.DataFrame
        DataFrame containing the  3  component values and the sleep stage labels.
    component_names : List[str]
        List containing the names of the PC axes to plot.
    stage_col : str, optional
        Name of the column indicating the labeled sleep stage (default is "stage").
    figure_save_path : str, optional
        Directory where the interactive figure will be saved (default is "decomposition").
    parameters : Dict[str, Dict[str, str]], optional
        Dictionary containing parameters such as color mapping for sleep stages (default is None).

    Returns:
    --------
    None:
        Saves an interactive HTML plot in the specified figure_save_path.
    """
    # Retrieve color map for sleep stages from parameters
    stage_color_map = parameters["stage_color_map"]

    # Unpack the component names
    x, y, z = component_names

    # Create 3D scatter plot using Plotly
    fig = go.Figure()

    # Plot each sleep stage separately with different colors
    for stage in stage_color_map.keys():
        stage_data = transformed_data_df[transformed_data_df[stage_col] == stage]

        # Main scatter plot with small markers for each stage
        fig.add_trace(
            go.Scatter3d(
                x=stage_data[x],
                y=stage_data[y],
                z=stage_data[z],
                mode="markers",
                marker=dict(
                    size=2,
                    color=stage_color_map[stage],
                    opacity=0.5,
                ),
                name=stage,
                showlegend=False,  # Hide this trace in the legend
            )
        )

        # Separate larger markers for the legend
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],  # Dummy points for the legend
                mode="markers",
                marker=dict(
                    size=12,  # Increase dot size for the legend
                    color=stage_color_map[stage],
                ),
                name=stage,
                showlegend=True,  # Show this trace in the legend
            )
        )

    # Update layout for the 3D plot
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

    # Save the interactive 3D plot as an HTML file
    fig.write_html(
        Path(figure_save_path) / f"{x}_{y}_{z}_3d_dimension_sleep_dynamics.html"
    )

    del fig


def plot_low_dimension_map(
    transformed_data_df: pd.DataFrame,
    component_names: List[str],
    stage_col: str,
    figure_save_path: str,
    parameters: Dict[str, Dict[str, str]],
) -> None:
    """
    Plots 2D and 3D scatter plots of principal components, colored by the labeled sleep stage.

    Args:
    -----
    transformed_data_df : pd.DataFrame
        DataFrame containing the PC values and the sleep stage labels.
    component_names : List[str]
        List containing the names of the PC axes to plot.
    stage_col : str
        Name of the column indicating the labeled sleep stage.
    figure_save_path : str
        Directory where the figures will be saved.
    parameters : Dict[str, Dict[str, str]]
        Dictionary containing parameters such as color mapping for sleep stages.

    Returns:
    --------
    None:
        Saves 2D and 3D plots as SVG and HTML files in the specified figure_save_path.
    """
    # Rename components for better readability
    transformed_data_df = transformed_data_df.rename(
        columns=lambda x: rename_pc(x) if "ca" in x else x
    )

    # Retrieve color map for sleep stages and rename PC axes
    stage_color_map = parameters["stage_color_map"]
    component_names = [rename_pc(x) for x in component_names]

    # Generate all 3D combinations of the principal components
    ordered_trip = list(itertools.combinations(np.arange(0, len(component_names)), 3))

    # Plot all 3D combinations of the PCs
    for trip in ordered_trip:
        plot_interactive_low_dimension_map(
            transformed_data_df=transformed_data_df,
            component_names=[component_names[i] for i in trip],
            stage_col=stage_col,
            figure_save_path=figure_save_path,
            parameters=parameters,
        )

    # Generate all 2D combinations of the principal components
    ordered_pairs = list(itertools.permutations(np.arange(0, len(component_names)), 2))

    # Plot all 2D combinations of the PCs
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
                s=2,  # Size of the markers
                alpha=0.5,  # Transparency for better visibility
            )

        # Add labels and title
        ax.set_xlabel(f"{component_names[pair[0]]} (PDU)")
        ax.set_ylabel(f"{component_names[pair[1]]} (PDU)")
        ax.set_title(
            f"2D Scatter Plot of {component_names[pair[0]]}-{component_names[pair[1]]} by Sleep Stage"
        )
        ax.legend(title="Sleep Stage", loc="upper left")

        # Save the 2D scatter plot as an SVG file
        fig.savefig(
            Path(figure_save_path)
            / f"{component_names[pair[0]]}_{component_names[pair[1]]}_2d_dimension_sleep_dynamics.svg"
        )
        plt.close()


def plot_component_weight_map(
    weights: pd.DataFrame,
    component_name: str,
    eeg_name_map: Dict[str, str],
    non_eeg_name_map: Dict[str, Dict[str, str]],
    channel_positions: pd.DataFrame,  # [x, y, z]
    title: str,
    save_path: Path,
    figure_columns: int = 8,
) -> plt.Figure:
    """
    Plots a feature map to visualize the spatial distribution of weights across EEG electrodes and non-EEG modalities
    for a given component.

    Args:
    -----
    weights : pd.DataFrame
        DataFrame containing the component weights for each feature (indexed by electrode or feature name).
    component_name : str
        Name of the component whose weight distribution is being plotted.
    eeg_name_map : Dict[str, str]
        Dictionary mapping EEG feature names to their more readable names for plotting.
    non_eeg_name_map : Dict[str, Dict[str, str]]
        Dictionary mapping non-EEG modality names to feature groupings (e.g., EOG, EMG).
    channel_positions : pd.DataFrame
        DataFrame containing the x, y, and z positions of each EEG electrode.
    title : str
        Title for the plot.
    save_path : Path
        Path where the figure will be saved.
    figure_columns : int, optional
        Number of columns to arrange the subplots (default is 8).

    Returns:
    --------
    plt.Figure:
        The generated matplotlib figure.
    """
    # Reset index of channel positions and ensure electrode names are in uppercase
    channel_positions = channel_positions.reset_index()
    channel_positions["electrode"] = channel_positions["electrode"].str.upper()

    # Create a dictionary of positions for EEG electrodes
    pos_dict = {}
    for _, row in channel_positions.iterrows():
        pos_dict[row["electrode"]] = [row["x"], row["y"], row["z"]]

    # Create an MNE montage for EEG electrode positions
    montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame="head")
    info = mne.create_info(
        ch_names=channel_positions["electrode"].to_list(), sfreq=1, ch_types="eeg"
    )
    info.set_montage(montage, match_case=False)

    # Identify non-EEG features (e.g., EOG, EMG)
    non_eeg_features = list(
        filter(
            lambda x: x.split("_")[0] not in channel_positions["electrode"].values,
            weights.index.values,
        )
    )

    # Add 'feature' and 'electrode' columns to weights DataFrame
    weights["feature"] = list(map(lambda x: "_".join(x.split("_")[1:]), weights.index))
    weights["electrode"] = list(map(lambda x: x.split("_")[0], weights.index))

    # Separate EEG and non-EEG weights
    non_eeg_weights = weights.loc[non_eeg_features, :]
    eeg_weights = weights.drop(index=non_eeg_features)

    # Determine the total number of subplots (EEG and non-EEG)
    total_subplots = 1 + len(eeg_name_map.keys()) + len(non_eeg_name_map.keys())
    rows = int(np.ceil(total_subplots / figure_columns))

    fig, ax = plt.subplots(rows, figure_columns, figsize=(3 * figure_columns, 3 * rows))

    colorbar_ax = ax[rows - 1, figure_columns - 1]
    ax[rows - 1, figure_columns - 1].set_xticks([])

    max_val = np.max(np.abs(weights[component_name]))  # Define symmetric color range
    tick_val = np.round(max_val * 0.6, decimals=3)
    # Plot EEG topomaps
    for i, feature in enumerate(eeg_name_map.keys()):
        x, y = np.divmod(i, figure_columns)
        feature_weights = eeg_weights[eeg_weights.feature == feature]
        feature_weights.set_index("electrode", inplace=True)

        ax[x, y].set_title(eeg_name_map[feature], fontsize=16)

        mne.viz.plot_topomap(
            data=feature_weights[component_name].values,
            pos=info,
            axes=ax[x, y],
            ch_type="eeg",
            vlim=[-max_val, max_val],
            show=False,
            extrapolate="local",
            sphere="auto",
            contours=4,
        )

    # Plot non-EEG feature weights using bar plots
    start_ind = len(eeg_name_map.keys())
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

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
            data=feature_weights,
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
        sns.despine(bottom=True, left=True, ax=ax[x, y])
        ax[x, y].set_title(f"{modality_group}", fontsize=16)
        ax[x, y].tick_params(axis="x", rotation=0)
        ax[x, y].set_xlabel("")
        ax[x, y].set_ylabel("")
        ax[x, y].axhline(
            y=0, color="gray", linestyle="--", label="0", alpha=0.5, linewidth=0.5
        )
        if i == 0:
            ax[x, y].set_yticks([-tick_val, tick_val])
        else:
            ax[x, y].set_yticks([])

        ax[x, y].set_ylim([-max_val, max_val])
        ax[x, y].tick_params(axis="y", rotation=90)
        ax[x, y].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

    # Create and add colorbar
    norm = mpl.colors.Normalize(vmin=-max_val, vmax=max_val)
    sns.despine(bottom=True, left=True, ax=colorbar_ax)
    colorbar = fig.colorbar(
        mappable=mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
        ax=colorbar_ax,
        use_gridspec=False,
        orientation="vertical",
        aspect=8,
        fraction=0.5,
        pad=0.5,
    )
    colorbar.ax.set_yticks([-tick_val, 0, tick_val])
    colorbar.ax.tick_params(labelsize=14, labelrotation=0)
    colorbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    colorbar.ax.set_xlabel("PDU", fontsize=16)
    sns.despine(bottom=True, left=True, ax=colorbar.ax)

    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path / f"{title.split(', ')[0]}_feature_map.svg")
    plt.close(fig)

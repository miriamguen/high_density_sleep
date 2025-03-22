""" Main script to load, clean, and preprocess feature matrices for decomposition analysis.
    This script loads all feature matrices, applies cleaning and standardization, and combines
    the data for further analysis.
"""

import os
import glob
from pathlib import Path
import yaml

from typing import List

import pandas as pd
import numpy as np

from decomposition_utils import (
    load_data_and_clean,
    decompose_all_patients,
    plot_low_dimension_map,
    plot_component_weight_map,
    rename_pc,
)


if __name__ == "__main__":
    """
    Main script to load, clean, and preprocess feature matrices for decomposition analysis.
    This script loads all feature matrices, applies cleaning and standardization, and combines
    the data for further analysis.
    """

    # Load analysis parameters from the YAML configuration file
    with open("analysis_code/parameters.yaml", "r") as file:
        PARAMETERS = yaml.safe_load(file)

    # Get the output directory and create necessary directories for saving processed data
    output_path = Path(PARAMETERS["OUTPUT_DIR"])
    feature_path = output_path / "features"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "processed_data", exist_ok=True)

    # Find all feature matrix files with the ".features.csv" extension
    feature_files = os.listdir(feature_path)

    # filter(
    #     lambda x: x.startswith("E"), glob.glob("**/*.features.csv", recursive=True)
    # )

    # Retrieve metadata columns from the parameters file
    metadata_columns: List[str] = PARAMETERS["METADATA_COLUMNS"]

    # If processed data already exists, load it from disk; otherwise, clean and standardize the data
    if os.path.exists(output_path / "processed_data" / "clean_data_all.parquet"):
        clean_data_all = pd.read_parquet(
            output_path / "processed_data" / "clean_data_all.parquet"
        )
    else:
        # Initialize lists to store cleaned data, means, stds, and quality metrics
        data_list: List[pd.DataFrame] = []
        means_list: List[pd.DataFrame] = []
        stds_list: List[pd.DataFrame] = []
        quality: List[pd.DataFrame] = []
        missing: dict = {}

        # Iterate over each feature matrix file, clean, and standardize the data
        for file in feature_files:
            # Load, clean, and standardize the feature data
            clean_data, means, stds, dropped_segments, clipped, missing[file] = (
                load_data_and_clean(
                    feature_path / file,
                    metadata_columns=metadata_columns,
                    parameters=PARAMETERS,
                )
            )

            # Append the cleaned data, means, and stds to the corresponding lists
            data_list.append(clean_data)
            means_list.append(
                means.reset_index(drop=False).set_index(["stage", "file"])
            )
            stds_list.append(stds.reset_index(drop=False).set_index(["stage", "file"]))

            # Record the percentage of clipped values and dropped segments
            quality.append(clipped.reset_index(drop=False).set_index(["index", "file"]))

        # Concatenate all cleaned data into a single DataFrame for further analysis
        clean_data_all = pd.concat(data_list).reset_index(drop=True)
        clean_data_all.to_parquet(
            output_path / "processed_data" / "clean_data_all.parquet"
        )

        # Save the calculated means, stds, and quality metrics
        means = pd.concat(means_list)
        means.to_csv(output_path / "processed_data" / "means.csv")

        stds = pd.concat(stds_list)
        stds.to_csv(output_path / "processed_data" / "stds.csv")

        quality = pd.concat(quality)
        quality.to_csv(output_path / "processed_data" / "clipped.csv")

        pd.DataFrame(missing).T.to_csv(output_path / "processed_data" / "missing.csv")

    # Perform decomposition (PCA/ICA) on the cleaned data
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
        figure_save_path=output_path / "decomposition",
        model_save_path=output_path / "decomposition" / "models",
        parameters=PARAMETERS,
    )


    transformed_data.to_parquet(
        output_path / "processed_data" / "transformed_data.parquet"
    )
    # Plot 2D and 3D low-dimensional projections of the PCA components
    plot_low_dimension_map(
        transformed_data_df=transformed_data,
        component_names=pc_names_ic,
        stage_col="stage",
        figure_save_path=output_path / "decomposition" / "pca",
        parameters=PARAMETERS,
    )

    # Plot 2D and 3D low-dimensional projections of the ICA components
    plot_low_dimension_map(
        transformed_data_df=transformed_data,
        component_names=ic_names,
        stage_col="stage",
        figure_save_path=output_path / "decomposition" / "ica",
        parameters=PARAMETERS,
    )

    # Load the electrode channel positions for plotting
    channel_positions = pd.read_csv(
        "Co-registered average positions.pos",
        header=None,
        delimiter="\t",
        names=["electrode", "y", "x", "z"],
    )
    channel_positions.set_index("electrode", inplace=True)

    # Plot component weight maps for the top PCA components
    for i, pc_name in enumerate(pc_names_ic):
        plot_component_weight_map(
            weights=pc_weights[[pc_name]],
            component_name=pc_name,
            eeg_name_map=PARAMETERS["eeg_name_map"],
            non_eeg_name_map=PARAMETERS["non_eeg_name_map"],
            channel_positions=channel_positions,
            title=f"{rename_pc(pc_name)}, ex: {pca_model.explained_variance_ratio_[i]:.2f}",
            save_path=output_path / "decomposition" / "pca",
        )

    # Compute the feature-to-IC weights matrix by multiplying PCA and ICA weights
    feature_to_ic_weights = pd.DataFrame(
        data=np.dot(a=pc_weights[pc_names_ic], b=ic_weights),
        index=pc_weights.index,
        columns=ic_weights.columns,
    )

    # Plot component weight maps for the ICA components
    for i, ic_name in enumerate(ic_names):
        plot_component_weight_map(
            weights=feature_to_ic_weights[[ic_name]],
            component_name=ic_name,
            eeg_name_map=PARAMETERS["eeg_name_map"],
            non_eeg_name_map=PARAMETERS["non_eeg_name_map"],
            channel_positions=channel_positions,
            title=f"{rename_pc(ic_name)}",
            save_path=output_path / "decomposition" / "ica",
        )

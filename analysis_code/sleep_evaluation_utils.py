""" This script include the code for all help functions used to analyse sleep parameters and agreement"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm
import pandas as pd
import numpy as np

import pingouin as pg
from yasa import sleep_statistics

import seaborn as sns
import matplotlib.pyplot as plt


def bland_altman_plot(
    data1: np.ndarray, data2: np.ndarray, measure: str, save_path: str
) -> None:
    """
    Creates and saves a Bland-Altman plot comparing two sets of measurements (data1 and data2).

    Parameters:
    -----------
    data1 : np.ndarray
        Measurements from the first method (e.g., hidden states).
    data2 : np.ndarray
        Measurements from the second method (e.g., manual sleep labels).
    measure : str
        The name of the measure being compared (e.g., "Sleep stage assignment").
    save_path : str
        File path to save the Bland-Altman plot.
    """
    # Calculate mean and difference between the two measurements
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)  # Mean difference
    sd = np.std(diff, axis=0)  # Standard deviation of differences

    # Define y-axis span for the plot
    y_span = np.max([np.std(mean) * 1.96, np.max(np.abs(diff))])
    y_span = y_span + y_span / 20

    # Create the Bland-Altman plot
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

    # Labeling the plot
    plt.xlabel("Mean of hidden states and sleep labels")
    plt.ylabel("Difference between hidden states and sleep labels")
    plt.ylim([-y_span, y_span])
    plt.title(f"Bland-Altman Plot for {measure}")
    plt.legend()
    plt.grid(True)

    # Save and close the plot
    plt.savefig(save_path)
    plt.close()


def calculate_icc(data1: pd.Series, data2: pd.Series) -> Tuple[float, pd.DataFrame]:
    """
    Calculates the Intraclass Correlation Coefficient (ICC) between two sets of measurements (data1 and data2).

    Parameters:
    -----------
    data1 : pd.Series
        Measurements from the first method (e.g., hidden states results).
    data2 : pd.Series
        Measurements from the second method (e.g., manual sleep labels).

    Returns:
    --------
    icc_value : float
        The calculated ICC value (specifically ICC2, two-way random effects, absolute agreement).
    icc_details : pd.DataFrame
        Detailed ICC results including other types of ICC.
    """
    # Create a DataFrame for ICC calculation
    df_icc = pd.DataFrame(
        {
            "Subject": data1.index.tolist() + data2.index.tolist(),
            "Rater": ["Cluster"] * len(data1) + ["Sleep label"] * len(data2),
            "Value": np.concatenate([data1.values, data2.values]),
        }
    )

    # Calculate Intraclass Correlation Coefficient (ICC)
    icc_results = pg.intraclass_corr(
        data=df_icc, targets="Subject", raters="Rater", ratings="Value"
    )

    # Extract ICC2 (Two-way random effects, absolute agreement)
    icc_value = icc_results.loc[icc_results["Type"] == "ICC2", "ICC"].values[0]
    return icc_value, icc_results


def get_sleep_measures_for_patient(
    labels: list,
    stage_map: Dict = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4},
    sampling_rate: float = 2.0,  # Samples per minute (e.g., 2 samples per minute)
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Computes sleep measures for both manual sleep labels and hidden states-derived sleep labels using YASA's
    `sleep_statistics` function.

    Parameters:
    -----------
    labels : list
        List of sleep stages obtained from an analysis.
    stage_map : Dict, optional
        Mapping of sleep stage labels to numerical values for YASA input. Default maps:
        {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}.
    sampling_rate : float, optional
        Number of samples per minute. Default is 2 samples per minute (i.e., 30-second epochs).

    Returns:
    --------
    Tuple[Dict[str, float], Dict[str, float]]:
        - sleep_measures: A dictionary of sleep measures calculated from the labeled sleep stages.
    """
    # Convert the manual and hidden states labels to YASA-compatible format
    labels = list(map(lambda x: stage_map[x], labels))

    # Define the sampling frequency in terms of hypnogram epochs (YASA uses samples per hour)
    sf_hyp = sampling_rate / 60  # Convert samples per minute to per hour

    # Calculate sleep statistics for both manual and hidden states-based labels using YASA
    sleep_measures = sleep_statistics(labels, sf_hyp)

    return sleep_measures


def compare_sleep_and_unsupervised_measures(
    sleep_measures_all: pd.DataFrame,
    unsupervised_measures_all: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Compares sleep measures derived from manual sleep stage annotations and unsupervised states.
    The function calculates and visualizes the differences between the two sets of measures by:
        - Creating scatter plots for direct comparison.
        - Performing ICC2 calculations.
        - Creating Bland-Altman plots for agreement analysis.

    Parameters:
    -----------
    sleep_measures_all : pd.DataFrame
        DataFrame containing sleep measures based on manually labeled sleep stages.
    unsupervised_measures_all : pd.DataFrame
        DataFrame containing sleep measures based on hidden states  minimally-supervised learning.
    save_path : where to save the results
    Returns:
    --------
    None:
        Results are saved as figures (scatter plots and Bland-Altman plots) and CSV files
        (ICC results and summary).
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path / "figures", exist_ok=True)
    icc2_result_summary: Dict[str, Dict] = {}

    # Ensure the 'Patient' and 'TRT' columns are not compared
    excluded_cols = ["Patient", "TRT"]

    # Iterate over the columns (measures) and perform paired tests
    for col in sleep_measures_all.columns:
        if col not in excluded_cols:
            # Extract and sort both the sleep measures and hidden states measures by index (patient)
            gold_standard_results = sleep_measures_all[col].sort_index()
            new_test_results = unsupervised_measures_all[col].sort_index()

            # Assert the indices of the two Series match before comparison
            assert all(
                gold_standard_results.index == new_test_results.index
            ), "Indices of the Series do not match."

            # Create scatter plot to visually compare the two measures
            plt.figure(figsize=(8, 6))
            plt.scatter(gold_standard_results, new_test_results, alpha=0.7)

            # Define plot boundaries with a small margin
            max_val = max(gold_standard_results.max(), new_test_results.max())
            min_val = min(gold_standard_results.min(), new_test_results.min())
            margin = max_val / 20

            # Add reference line (identity line) to show the perfect agreement
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                lw=2,
            )

            # Set plot labels and title
            plt.xlabel(f"{col} by sleep labels")
            plt.ylabel(f"{col} by GHMM")
            plt.xlim((min_val - margin, max_val + margin))
            plt.ylim((min_val - margin, max_val + margin))
            plt.title(f"Comparing {col} estimation (Visual Vs. GHMM)")
            plt.grid(True)

            # Save the scatter plot
            os.makedirs(save_path / "figures", exist_ok=True)
            plt.savefig(save_path / "figures" / f"{col}_scatter_plot.svg")
            plt.close()

            # Calculate Intraclass Correlation Coefficient (ICC)
            icc_value, icc_details = calculate_icc(
                new_test_results, gold_standard_results
            )
            icc2_result_summary[col] = (
                icc_details.set_index("Type").loc["ICC2", :].to_dict()
            )
            print(f"Intraclass Correlation Coefficient (ICC) for {col}:")
            print(f"ICC Value (Type ICC2): {icc_value:.3f}\n")

            # Save the detailed ICC results as CSV
            icc_details.to_csv(save_path / f"{col}_icc_results.csv", index=False)

            # Create Bland-Altman Plot for agreement analysis
            bland_altman_plot(
                new_test_results.values,
                gold_standard_results.values,
                measure=col,
                save_path=save_path / "figures" / f"{col}_bland_altman_plot.svg",
            )

    # Save the ICC2 summary results as CSV
    pd.DataFrame(icc2_result_summary).to_csv(save_path / "icc2_result_summary.csv")

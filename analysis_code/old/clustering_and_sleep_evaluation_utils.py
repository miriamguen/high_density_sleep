""" This script include the code for all help functions used to analyse the PC and IC space in relation to sleep parameters of the visual labels"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm
import pandas as pd
import numpy as np

import pingouin as pg
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    v_measure_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
)

from yasa import sleep_statistics
import seaborn as sns
import matplotlib.pyplot as plt


def map_cluster_labels(
    sleep_labels: np.ndarray, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    Aligns the cluster labels with the true sleep stage labels by maximizing the match between clusters and true labels.

    Parameters:
    -----------
    sleep_labels : np.ndarray
        Array of true sleep stage labels.
    cluster_labels : np.ndarray
        Array of predicted cluster labels.

    Returns:
    --------
    np.ndarray:
        Mapped cluster labels that best align with the true sleep stage labels.
    """
    # Create a confusion matrix between sleep stage labels and cluster labels
    cm = pd.crosstab(
        pd.Series(sleep_labels, name="Sleep stage"),
        pd.Series(cluster_labels, name="Cluster"),
    )

    # Use the Hungarian algorithm to maximize the alignment between cluster and true labels
    row_ind, col_ind = linear_sum_assignment(-cm)

    if len(set(sleep_labels)) < len(set(cluster_labels)):
        # if there are more clusters then labels some clusters will miss a label
        # identify unlabeled cluster
        unlabeled_clusters = list(filter(lambda x: x not in col_ind, cm.columns.values))
        # label them with the maximal class
        for cluster in unlabeled_clusters:
            col_ind = np.append(col_ind, cluster)
            row_ind = np.append(row_ind, cm[cluster].argmax())

    # Create a mapping from cluster labels to true sleep stage labels
    mapping = {
        cluster_label: true_label for cluster_label, true_label in zip(col_ind, row_ind)
    }
    # Apply the mapping to the cluster labels
    mapped_labels = np.vectorize(mapping.get)(cluster_labels)

    return mapped_labels, mapping


def evaluate_cluster_labels(
    sleep_labels: np.ndarray,
    sleep_labels_by_cluster: np.ndarray,
    data: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluates the clustering performance by computing several metrics such as Silhouette Score,
    Davies-Bouldin Index, precision, recall (sensitivity), and specificity for each sleep stage.
    The function also generates a confusion matrix arranged by sleep stage order.

    Parameters:
    -----------
    sleep_labels : np.ndarray
        The true sleep stage labels.
    sleep_labels_by_cluster : np.ndarray
        The cluster labels aligned with sleep stages.
    data : np.ndarray
        The data used for clustering (e.g., principal components or features).

    Returns:
    --------
    result : Dict[str, float]
        A dictionary containing clustering metrics and precision, recall, specificity for each sleep stage.
    confusion_matrix : pd.DataFrame
        Confusion matrix arranged by sleep stage order.
    """

    # Define the correct order of sleep stages
    stage_order = ["W", "R", "N1", "N2", "N3"]

    # Ensure sleep labels are only within the allowed sleep stages
    assert set(np.unique(sleep_labels)).issubset(
        stage_order
    ), f"sleep_labels contains invalid stages. Allowed stages: {stage_order}"

    assert set(np.unique(sleep_labels_by_cluster)).issubset(
        stage_order
    ), f"sleep_labels_by_cluster contains invalid stages. Allowed stages: {stage_order}"

    # Compute unsupervised clustering evaluation metrics
    result = {
        "Silhouette Score": silhouette_score(data, sleep_labels_by_cluster),
        "Davies-Bouldin Index": davies_bouldin_score(data, sleep_labels_by_cluster),
        "v_measure_score": v_measure_score(sleep_labels, sleep_labels_by_cluster),
        "adjusted_rand_score": adjusted_rand_score(
            sleep_labels, sleep_labels_by_cluster
        ),
        "normalized_mutual_info_score": normalized_mutual_info_score(
            sleep_labels, sleep_labels_by_cluster
        ),
        "overall_accuracy": accuracy_score(sleep_labels, sleep_labels_by_cluster),
    }

    # Generate confusion matrix
    confusion_matrix = pd.crosstab(
        pd.Series(sleep_labels, name="Sleep stage"),
        pd.Series(sleep_labels_by_cluster, name="Cluster"),
        dropna=False
    )

    stage_order = list(
        filter(lambda x: x in confusion_matrix.columns.values, stage_order)
    )
    # Re-arrange by stage order
    confusion_matrix = confusion_matrix.loc[stage_order, stage_order]

    total_samples = confusion_matrix.sum().sum()

    # Iterate over each sleep stage to calculate precision, recall (sensitivity), and specificity
    for stage in stage_order:
        total_stage = confusion_matrix.loc[stage, :].sum()  # TP + FN
        total_stage_cluster_labels = confusion_matrix.loc[:, stage].sum()  # TP + FP
        TP = confusion_matrix.loc[stage, stage]
        FN = total_stage - TP
        FP = total_stage_cluster_labels - TP
        TN = total_samples - TP - FN - FP

        # Calculate precision, recall (sensitivity), and specificity
        result[f"sensitivity {stage}"] = (
            TP / (TP + FN) if (TP + FN) > 0 else 0 if TP > 0 else np.nan
        )
        result[f"specificity {stage}"] = (
            TN / (TN + FP) if (TN + FP) > 0 else 0 if TN > 0 else np.nan
        )
        result[f"precision {stage}"] = (
            TP / (TP + FP) if (TP + FP) > 0 else 0 if TN > 0 else np.nan
        )
        result[f"f1  {stage}"] = (
            2
            * (result[f"precision {stage}"] * result[f"sensitivity {stage}"])
            / (result[f"precision {stage}"] + result[f"sensitivity {stage}"])
        )

    return result, confusion_matrix


def cluster_analysis(
    transformed_data_: pd.DataFrame,
    feature_names: List[str],
    label_col: str,
    sleep_label_txt_map: dict,
    n_clusters: list,
    result_path: Path,
) -> pd.DataFrame:
    """
    Performs KMeans clustering with varying cluster sizes (K=5 to K=10) and evaluates the clustering results
    using various metrics. It also generates a leave-one-out (LOO) analysis for each patient, producing metrics
    such as sensitivity and specificity distributions across patients. Confusion matrices and other performance
    plots are generated and saved.

    Parameters:
    -----------
    transformed_data_ : pd.DataFrame
        DataFrame containing the transformed data along with sleep stages and other labels.
    feature_names : List[str]
        List of column names used as features for clustering.
    label_col : str
        The name of the label column with numerical values.
    sleep_label_txt_map: dict
        A mapping from numerical to textual sleep stage labels.
    result_path : Path
        Path to the directory where clustering results (figures, metrics) will be saved.

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing the transformed data with cluster labels added.
    """
    transformed_data = transformed_data_.copy(deep=True)
    transformed_data[feature_names] = transformed_data[feature_names] /transformed_data[feature_names].std()
    # Create the result directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)

    # Get the list of unique patients
    patient_list = list(set(transformed_data["patient"]))

    # Iterate over different values of K (number of clusters)
    for k in n_clusters:
        train_results = {}
        test_results = {}
        aggregated_labels = []

        # Perform leave-one-out analysis over patients
        print(f"working on k={k}")
        for patient in tqdm(patient_list):
            train = transformed_data.query(f"patient != '{patient}'")
            test = transformed_data.query(f"patient == '{patient}'")

            # Apply KMeans clustering on the training set
            kmeans = KMeans(n_clusters=k, random_state=42).fit(train[feature_names])
            train_cluster_labels = kmeans.predict(train[feature_names])
            train_labels = train[label_col]

            # Map cluster labels to sleep labels using the training set
            train_sleep_labels_by_cluster, cluster_to_label_map = map_cluster_labels(
                train_labels, train_cluster_labels
            )

            # Map to textual labels for evaluation
            train_labels = [sleep_label_txt_map[x] for x in train_labels]
            train_sleep_labels_by_cluster = [
                sleep_label_txt_map[x] for x in train_sleep_labels_by_cluster
            ]

            # Evaluate clustering on the training set
            train_results[patient], _ = evaluate_cluster_labels(
                train_labels, train_sleep_labels_by_cluster, train[feature_names]
            )

            # Apply the cluster model on the test set
            test_labels = [sleep_label_txt_map[x] for x in test[label_col]]
            test_cluster_labels = list(
                map(
                    lambda x: cluster_to_label_map[x],
                    kmeans.predict(test[feature_names]),
                )
            )
            test_sleep_labels_by_cluster = [
                sleep_label_txt_map[x] for x in test_cluster_labels
            ]

            # Evaluate clustering on the test set
            test_results[patient], _ = evaluate_cluster_labels(
                test_labels, test_sleep_labels_by_cluster, test[feature_names]
            )

            # Collect aggregated test labels for all patients
            test = test.assign(
                test_labels=test_labels,
                test_cluster_labels=test_cluster_labels,
                test_sleep_labels_by_cluster=test_sleep_labels_by_cluster,
            )

            aggregated_labels.append(
                test[
                    [
                        "patient",
                        label_col,
                        "test_labels",
                        "test_cluster_labels",
                        "test_sleep_labels_by_cluster",
                    ]
                    + feature_names
                ]
            )

        # Convert results to DataFrame for easier handling
        train_results = pd.DataFrame(train_results)
        test_results = pd.DataFrame(test_results)
        aggregated_labels = pd.concat(aggregated_labels)

        # Evaluate overall test performance and plot confusion matrix
        overall_test_results, confusion_matrix = evaluate_cluster_labels(
            aggregated_labels["test_labels"],
            aggregated_labels["test_sleep_labels_by_cluster"],
            aggregated_labels[feature_names],
        )

        # Plot and save the confusion matrix
        title = f"Test K = {k} cluster vs manual labels"
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d")
        plt.title(title)
        fig.savefig(result_path / f"{title}.svg")
        plt.close()
        confusion_matrix.to_csv(result_path / f"{title}.csv")
        stage_color_map = {
            "W": "lightsalmon",
            "R": "indianred",
            "N1": "pink",
            "N2": "lightblue",
            "N3": "royalblue",
        }

        # # Plot sensitivity and specificity distribution over patients
        # fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey="row")

        # # Sensitivity distribution
        # sns.violinplot(
        #     data=test_results.T.filter(like="sensitivity").rename(
        #         columns=lambda x: x.replace("sensitivity ", "")
        #     ),
        #     inner="point",
        #     ax=ax[0],
        #     palette=stage_color_map,
        # )
        # ax[0].set_title(f"Sensitivity Distribution per Sleep Stage (K={k})")
        # ax[0].set_ylabel("Sensitivity")
        # ax[0].set_xlabel("Sleep Stage")
        # ax[0].set_ylim([-0.1, 1.1])

        # # Specificity distribution
        # sns.violinplot(
        #     data=test_results.T.filter(like="precision").rename(
        #         columns=lambda x: x.replace("precision ", "")
        #     ),
        #     inner="point",
        #     ax=ax[1],
        #     palette=stage_color_map,
        # )
        # ax[1].set_title(f"Precision Distribution per Sleep Stage (K={k})")
        # ax[1].set_ylabel("Precision")
        # ax[1].set_xlabel("Sleep Stage")

        # plt.tight_layout()
        # fig.savefig(result_path / f"Performance_Distribution_K{k}.svg")
        # plt.close()

        # Save leave-one-out performance results
        train_results.to_csv(result_path / f"{k}_loo_cluster_metrics_train.csv")
        test_results.to_csv(result_path / f"{k}_loo_cluster_metrics_test.csv")
        aggregated_labels.to_csv(result_path / f"{k}_aggregated_labels.csv")

    return transformed_data


def bland_altman_plot(
    data1: np.ndarray, data2: np.ndarray, measure: str, save_path: str
) -> None:
    """
    Creates and saves a Bland-Altman plot comparing two sets of measurements (data1 and data2).

    Parameters:
    -----------
    data1 : np.ndarray
        Measurements from the first method (e.g., clustering).
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
    plt.xlabel("Mean of clusters and sleep labels")
    plt.ylabel("Difference between clusters and sleep labels")
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
        Measurements from the first method (e.g., clustering results).
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
    Computes sleep measures for both manual sleep labels and cluster-derived sleep labels using YASA's
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
    # Convert the manual and cluster labels to YASA-compatible format
    labels = list(map(lambda x: stage_map[x], labels))
    labels = list(map(lambda x: stage_map[x], labels))

    # Define the sampling frequency in terms of hypnogram epochs (YASA uses samples per hour)
    sf_hyp = sampling_rate / 60  # Convert samples per minute to per hour

    # Calculate sleep statistics for both manual and cluster-based labels using YASA
    sleep_measures = sleep_statistics(labels, sf_hyp)

    return sleep_measures


def compare_sleep_and_cluster_measures(
    sleep_measures_all: pd.DataFrame, cluster_measures_all: pd.DataFrame
) -> None:
    """
    Compares sleep measures derived from manual sleep stage annotations and unsupervised clusters.
    The function calculates and visualizes the differences between the two sets of measures by:
        - Creating scatter plots for direct comparison.
        - Performing ICC2 calculations.
        - Creating Bland-Altman plots for agreement analysis.

    Parameters:
    -----------
    sleep_measures_all : pd.DataFrame
        DataFrame containing sleep measures based on manually labeled sleep stages.
    cluster_measures_all : pd.DataFrame
        DataFrame containing sleep measures based on clusters from unsupervised learning.

    Returns:
    --------
    None:
        Results are saved as figures (scatter plots and Bland-Altman plots) and CSV files
        (ICC results and summary).
    """
    icc2_result_summary: Dict[str, Dict] = {}

    # Ensure the 'Patient' and 'TRT' columns are not compared
    excluded_cols = ["Patient", "TRT"]

    # Iterate over the columns (measures) and perform paired tests
    for col in sleep_measures_all.columns:
        if col not in excluded_cols:
            # Extract and sort both the sleep measures and cluster measures by index (patient)
            gold_standard_results = sleep_measures_all[col].sort_index()
            new_test_results = cluster_measures_all[col].sort_index()

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
            plt.xlabel(f"{col} assessed with sleep labels")
            plt.ylabel(f"{col} assessed with clusters")
            plt.xlim((min_val - margin, max_val + margin))
            plt.ylim((min_val - margin, max_val + margin))
            plt.title(f"Comparing {col} estimation (Manual Vs. Unsupervised)")
            plt.grid(True)

            # Save the scatter plot
            os.makedirs("sleep_measures/figures", exist_ok=True)
            plt.savefig(f"sleep_measures/figures/{col}_scatter_plot.svg")
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
            icc_details.to_csv(f"sleep_measures/{col}_icc_results.csv", index=False)

            # Create Bland-Altman Plot for agreement analysis
            bland_altman_plot(
                new_test_results.values,
                gold_standard_results.values,
                measure=col,
                save_path=f"sleep_measures/figures/{col}_bland_altman_plot.svg",
            )

    # Save the ICC2 summary results as CSV
    pd.DataFrame(icc2_result_summary).to_csv("sleep_measures/icc2_result_summary.csv")

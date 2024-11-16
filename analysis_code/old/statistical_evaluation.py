"""This scrip contains the functions used for statistical analysis in thi project mainly:
    1. Testing for statistical agreement between the labeled clusters and each sleep stage accounting foe patient as a random effect
    2. Analyzing the linear relationship between the main PC and IC values and sleep stages
    3. Test the correlation between the patient ex in each IC/PC and overall sleep quality measures [SE, WASO, ]
    4. Testing the agreement between the overnight parameters estimated from the unsupervised measures and the sleep labels
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytensor as pt
import pymc as pm
import bambi as bmb


def prepare_data(
    df, patient="patient", cluster_col="cluster", sleep_label_col="sleep_label"
):
    # Convert categorical variables to categorical data types
    df[patient] = df[patient].astype("category")
    df[cluster_col] = df[cluster_col].astype("category")
    df[sleep_label_col] = df[sleep_label_col].astype("category")

    # Encode categorical variables as integer codes for modeling
    df["patient_code"] = df[patient].cat.codes
    df["cluster_code"] = df[cluster_col].cat.codes
    df["sleep_label_code"] = df[sleep_label_col].cat.codes

    # Map codes to labels for future reference
    patient_mapping = dict(enumerate(df[patient].cat.categories))
    cluster_mapping = dict(enumerate(df[cluster_col].cat.categories))
    cluster_mapping = dict(enumerate(df[sleep_label_col].cat.categories))

    # Number of unique patients, clusters, and sleep stages
    n_patients = df["patient_code"].nunique()
    n_clusters = df["cluster_code"].nunique()
    n_sleep_stages = df["sleep_label_code"].nunique()

    return (
        df,
        patient_mapping,
        cluster_mapping,
        cluster_mapping,
        n_patients,
        n_clusters,
        n_sleep_stages,
    )


if __name__ == "__main__":
    DATA_PATH = Path("clustering")
    SAVE_PATH = Path("statistical_analysis")

    # load the data
    df = pd.read_csv(DATA_PATH / "transformed_data_with_cluster_labels.csv")

    (
        df,
        patient_mapping,
        cluster_mapping,
        cluster_mapping,
        n_patients,
        n_clusters,
        n_sleep_stages,
    ) = prepare_data(
        df, patient="patient", cluster_col="cluster_labels_5", sleep_label_col="stage"
    )
    print(n_patients)

    cluster_stage_model = pm.Model()





    with cluster_stage_model:
        # Choose an appropriate sampler
        trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=2)

    pm.traceplot(trace, var_names=['intercept', 'cluster_coeffs', 'sigma_patient'])
    pm.summary(trace, var_names=['intercept', 'cluster_coeffs', 'sigma_patient'])

    # %% 1. Testing for statistical agreement between the labeled clusters and each sleep stage accounting foe patient as a random effect

    # %% 2. Analyzing the linear relationship between the main PC and IC values and sleep stages
    # %% 3. Test the correlation between the patient ex in each IC/PC and overall sleep quality measures [SE, WASO, ]
    # %% 4. Testing the agreement between the overnight sleep quality measures estimated from the unsupervised measures and the sleep labels
    # copy from long script

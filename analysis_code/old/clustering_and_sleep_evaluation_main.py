import os
import yaml
from pathlib import Path
import pandas as pd


from clustering_and_sleep_evaluation_utils import cluster_analysis

if __name__ == "__main__":
    with open("analysis_code/parameters.yaml", "r") as file:
        PARAMETERS = yaml.safe_load(file)

    output_dir = Path(PARAMETERS["OUTPUT_DIR"])
    # load the PC transformed data
    data = pd.read_parquet(output_dir / "processed_data" / "transformed_data.parquet")
    label_col = "num"
    clustering_path = output_dir / "clustering"
    statistical_model_path = output_dir / "statistical_models"
    os.makedirs(clustering_path, exist_ok=True)
    os.makedirs(statistical_model_path, exist_ok=True)
    os.makedirs(clustering_path / "ica", exist_ok=True)
    os.makedirs(clustering_path / "pca", exist_ok=True)
    pc_names = ["pca0", "pca1", "pca2", "pca3", "pca4", "pca5"]
    ic_names = ["ica0", "ica1", "ica2", "ica3", "ica4", "ica5"]

    # analyse the relationship between ICs and sleep stages
    import numpy as np

    patients = data["patient"].unique()
    patient_data = data[data["patient"] == patients[0]]

    sleep_label_txt_map = {
        v: k for k, v in PARAMETERS["sleep_stage_hypno_values"].items()
    }

    patients = data["patient"].unique()

    patient_data = data[data["patient"] == patients[0]]

    # cluster analysis using k-means

    cluster_analysis(
        data,
        label_col=label_col,
        feature_names=pc_names,
        sleep_label_txt_map=sleep_label_txt_map,
        n_clusters=np.arange(5, 26),  # range(5, 6)
        result_path=clustering_path / "pca",
    )

    cluster_analysis(
        data,
        label_col=label_col,
        feature_names=ic_names,
        sleep_label_txt_map=sleep_label_txt_map,
        n_clusters=np.arange(5, 26),  # range(5, 20)
        result_path=clustering_path / "ica",
    )

    print("ready for the next analysis")

    #

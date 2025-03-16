"load the pre trained models (pca/ica/ghmm) and predict the states of the data"

import os
import pickle
from typing import List
import numpy as np  # noqa
import pandas as pd  # noqa
import yaml  # noqa
import joblib
from pathlib import Path  # noqa
import pickle
from decomposition_utils import load_data_and_clean

from hmm_utils import evaluate

with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)

data_dir = Path(PARAMETERS["DATA_DIR"])
model_version = "output_short"
data_vresion = "output_short_overlap"
pca_ica = "ica"
label_col = "stage"

pca_wights = pd.read_csv(
    data_dir / model_version / "decomposition" / "models" / "pca_weights.csv",
    index_col=0,
)

ica_wights = pd.read_csv(
    data_dir / model_version / "decomposition" / "models" / "pca_to_ica_weights.csv",
    index_col=0,
)

with open(
    data_dir
    / model_version
    / "hmm"
    / f"{pca_ica}_{ica_wights.shape[0]}"
    / "common_model"
    / "model.pkl",
    "rb",
) as file:
    hmm_model = pickle.load(file)

with open(
    data_dir
    / model_version
    / "hmm"
    / f"{pca_ica}_{ica_wights.shape[0]}"
    / "common_model"
    / "model_mapping.pkl",
    "rb",
) as file:
    state_to_stage = pickle.load(file)

data_path = data_dir / data_vresion / "features"

os.makedirs(data_dir / data_vresion / "predicted_states", exist_ok=True)

metadata_columns: List[str] = PARAMETERS["METADATA_COLUMNS"]

for file in data_path.glob("*.csv"):
    clean_data, means, stds, dropped_segments, clipped, missing = load_data_and_clean(
        file, metadata_columns, PARAMETERS
    )
    metadata = clean_data[metadata_columns]
    pca_data = pd.DataFrame(
        data=np.dot(
            a=clean_data.drop(columns=metadata_columns),
            b=pca_wights.iloc[:, 0 : len(ica_wights)],
        ),
        index=clean_data.index,
        columns=pca_wights.columns[: len(ica_wights)],
    )

    ica_data = pd.DataFrame(
        data=np.dot(a=pca_data, b=ica_wights),
        index=pca_data.index,
        columns=ica_wights.columns,
    )

    patient_data = pd.concat(
        [
            metadata,
            pca_data,
            ica_data,
        ],
        axis=1,
    )

    lengths = patient_data.groupby("patient").size().tolist()
    bic, aic, adjusted_log_likelihood, accuracy, kappa, results = evaluate(
        hmm_model,
        state_to_stage,
        ica_data,
        labels=metadata[label_col],
        lengths=lengths,
        return_samples=True,
    )

    patient_data = pd.concat(
        [
            patient_data,
            results,
        ],
        axis=1,
    )

    patient_data.to_csv(file.parent.parent / "predicted_states" / file.name)

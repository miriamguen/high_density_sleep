import pandas as pd
import yaml
from pathlib import Path
import os
from decomposition_utils import (
    plot_component_weight_map,
)


# Load parameters
with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)

    # Load the electrode channel positions for plotting
    channel_positions = pd.read_csv(
        Path(PARAMETERS["DATA_DIR"]) / "Co-registered average positions.pos",
        header=None,
        delimiter="\t",
        names=["electrode", "y", "x", "z"],
    )
    channel_positions.set_index("electrode", inplace=True)

output_dir = Path(PARAMETERS["OUTPUT_DIR"])
electrodes = PARAMETERS["select_channels"]
if electrodes == "None":
    electrodes = None

data = (
    pd.read_parquet(output_dir / "processed_data" / "clean_data_all.parquet")
    .set_index(["patient", "time_from_onset"])
    .drop(columns=["time", "epoch_length", "num"])
)

metadata_columns = ["stage"]
stage_data = data.groupby("stage").mean()
stages = stage_data.index
os.makedirs(output_dir / "stage_maps", exist_ok=True)
for stage in stages:
    weights = stage_data.loc[[stage], :].T
    plot_component_weight_map(
        weights=weights,
        component_name=stage,
        eeg_name_map=PARAMETERS["eeg_name_map"],
        non_eeg_name_map=PARAMETERS["non_eeg_name_map"],
        channel_positions=channel_positions,
        title=f"{stage.title()} mean feature map",
        save_path=output_dir / "stage_maps",
        electrodes=electrodes
    )

ica_pca = "ica"
folder = list(filter(lambda x: x.startswith(ica_pca), os.listdir(output_dir / "hmm")))[
    0
]
hmm_dir = output_dir / "hmm" / folder / "common_model"
os.makedirs(hmm_dir / "state_maps", exist_ok=True)

labels = pd.read_csv(hmm_dir / "state_assignment_results.csv", index_col=0).set_index(
    ["patient", "time_from_onset"]
)

labels["hidden_states"] = pd.Series(
    data=[
        f"{n} ({s})"
        for n, s in zip(labels["hidden_states"], labels["hidden_states_mapped"])
    ],
    index=data.index,
)

n_states = labels["hidden_states"].nunique()

labels = labels[labels[list(map(str, range(n_states)))].max(axis=1) > 0.99]
data = data.loc[labels.index, :]

data.loc[labels.index, "hidden_states"] = labels["hidden_states"]
data.drop(columns=["stage"], inplace=True)

state_data = data.groupby("hidden_states").mean()
states = state_data.index


for state in states:
    weights = state_data.loc[[state], :].T
    plot_component_weight_map(
        weights=weights,
        component_name=state,
        eeg_name_map=PARAMETERS["eeg_name_map"],
        non_eeg_name_map=PARAMETERS["non_eeg_name_map"],
        channel_positions=channel_positions,
        title=f"{state.title()} mean feature map",
        save_path=hmm_dir / "state_maps",
        electrodes=electrodes,
        color_map="PRGn",
    )

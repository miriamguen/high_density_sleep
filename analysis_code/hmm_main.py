# main.py
import os
from matplotlib import pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import pickle


from hmm_utils import (
    create_state_profile,
    estimate_transition_probabilities,
    train_and_map,
    evaluate,
    fit_and_score_cv,
    plot_evaluation_metrics,
    select_best,
    plot_hidden_state_stage_distribution,
    plot_parameter_means_and_ci,
    plot_transition_graph,
    plot_state_time_histogram,
)

from sleep_evaluation_utils import (
    get_sleep_measures_for_patient,
    compare_sleep_and_unsupervised_measures,
)

# Load parameters
with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)

output_dir = Path(PARAMETERS["OUTPUT_DIR"])
data = pd.read_parquet(output_dir / "processed_data" / "transformed_data.parquet")
label_col = "stage"
stage_color_map = PARAMETERS["stage_color_map"]

hmm_path = output_dir / "hmm"
os.makedirs(hmm_path / "ica", exist_ok=True)


def add_time(df):
    df["time"] = df["time_from_onset"] / df["time_from_onset"].max()
    return df


# add time for the analysis of state- time
data = (
    data.groupby("patient")
    .apply(add_time, include_groups=False)
    .reset_index(drop=False)
)

# use the first 5 ICs, remove the last as it seems less relevant to stages
ic_names = ["ica0", "ica1", "ica2", "ica3", "ica4", "ica5"]

results = []

# scan the state range from 2 to 25 using leave one out cross validation
for n_states in tqdm(range(2, 26)):
    results.append(fit_and_score_cv(data, ic_names, label_col, n_states, scale=False))

# # save the cross validation results
results = pd.concat(results, axis=0)
results = results.assign(data="results_ic")
results.to_csv(hmm_path / "ica" / "ica_n_state_scan.csv")
results = pd.read_csv(hmm_path / "ica" / "ica_n_state_scan.csv")

# use the CV results to select the model using the BIC score, by evaluating a consistent reduction, yet not a label fit,
# to keep selection it unsupervised
best_groups = select_best(results.drop(columns=["kappa"]))

# plot the evaluation results and the selected state
plot_evaluation_metrics(
    results_df=results,
    best_states=[best_groups],
    save_path=hmm_path / "ica" / "ica_6_best_n_states.svg",
)

# %% Shared state analysis:
# 1. Train a model over patients and evaluate overall
common_path = hmm_path / "ica" / "common_model"
os.makedirs(common_path, exist_ok=True)
lengths = data.groupby("patient").size().tolist()
model_all, state_to_stage = train_and_map(
    data[ic_names], data[label_col], lengths, best_groups
)

bic, adjusted_log_likelihood, accuracy, kappa, results = evaluate(
    model_all,
    state_to_stage,
    data[ic_names],
    labels=data[label_col],
    lengths=lengths,
    return_samples=True,
)

results["time"] = data.time.values
results.to_csv(common_path / "state_assignment_results.csv")
model_metrics = pd.Series(
    {
        "bic": bic,
        "adjusted_log_likelihood": adjusted_log_likelihood,
        "accuracy": accuracy,
        "kappa": kappa,
    }
)
model_metrics.to_csv(common_path / "model_metrics.csv")

state_info = pd.DataFrame(state_to_stage, index=["mapped_states_all"]).T
state_info["highest_rate"] = results.groupby("hidden_states")["time"].apply(
    lambda x: np.histogram(x, bins=8)[0].argmax()
)
state_info["state_ids_all"] = [f"{k} ({v})" for k, v in state_to_stage.items()]
state_info = state_info.reset_index().rename(columns={"index": "hidden_state"})
state_info = state_info.sort_values(
    ["highest_rate", "mapped_states_all"], ascending=[True, False]
)

# state_order = [4,5,2,7,0,1,6,3]
# 2. Plot the sleep state distribution in each hidden state
state_label_counts = plot_hidden_state_stage_distribution(
    model_all,
    results,
    stage_color_map,
    state_info["state_ids_all"].values,
    save_path=common_path / "state_proportion_in_hidden.svg",
    alpha_value=0.7,
    title=None,  # f"Overall hidden state assignment, \nKappa={kappa:0.2f}, Accuracy={accuracy:0.2f}",
)
results["state_ids"] = [f"{k} ({state_to_stage[k]})" for k in results.hidden_states]
state_counts = (
    pd.DataFrame(state_label_counts)
    .fillna(0)
    .rename(columns=lambda x: int(x[0]), index=lambda x: f"{x}_count")
    .reset_index()
    .rename(columns={"index": "parameter"})
)
summary_df = create_state_profile(model_all, results, state_to_stage, ic_names)
summary_df = pd.concat([state_counts, summary_df], axis=0)
summary_df.to_csv(common_path / "model_summary_info.csv")

# look at the IC value distribution of each hidden state, of the different labeled sleep stages within each hidden - to asses for similarity
state_order = state_info["state_ids_all"].values
for i, ic in enumerate(ic_names):
    fig, ax = plt.subplots(figsize=(4, 10))
    ax.axvline(x=0, color="gray", linestyle="--", label="0", alpha=0.75, linewidth=1)
    sns.violinplot(
        x=data[ic],
        y=results["state_ids"],
        hue=results["hidden_states_mapped"],
        palette=PARAMETERS["stage_color_map"],
        density_norm="count",
        cut=0,
        ax=ax,
        inner="quart",
        order=state_info["state_ids_all"].values,
        legend=False,
    )

    ax.set_yticks(list(range(len(state_order))))
    ax.set_yticklabels(
        list(state_order),
        rotation=90,
        va="center",
        fontsize=12,
    )

    max_val = max(-data[ic].min(), data[ic].max()) * 1.1
    ax.set_xlabel(f"{ic.split('a')[0].upper()} {i+1}")
    ax.set_ylabel("")
    ax.set_xlim(-max_val, max_val)
    fig.savefig(common_path / f"{ic}_hidden_state_empirical_distribution.svg")
    plt.close()


# 3. Plot the feature distribution in each state
hidden_stage_color_map = {
    f"{k} ({v})": PARAMETERS["stage_color_map"][v] for k, v in state_to_stage.items()
}


plot_parameter_means_and_ci(
    model_all,
    feature_names=ic_names,
    stage_color_map=hidden_stage_color_map,
    state_order=list(reversed(list(state_info["state_ids_all"].values))),
    save_path=common_path / "state_parameter_distribution.svg",
)

plot_state_time_histogram(
    results,
    bins=50,
    stage_color_map=stage_color_map,
    state_order=state_info["hidden_state"].values,
    save_path=common_path / "state_time_distribution.svg",
)

# 4. Plot the cross patient state transition probability matrix
transition_probs = pd.DataFrame(
    data=model_all.transmat_,
    columns=range(model_all.n_components),
    index=range(model_all.n_components),
)
# state_order_network = [3, 4, 7, 5, 0, 2, 6, 1]
G = plot_transition_graph(
    transition_probs,
    common_path / "transition_prob.svg",
    manual=False,
    state_map=state_to_stage,
)

# %% use the graph to identify common transition paths across patients
# from collections import Counter
# from itertools import product

# transition_info = []
# patients = []
# state = ""
# run_patient = ""
# for new_state, patient in zip(results["state_ids"].values, data["patient"]):
#     if patient == run_patient:
#         if state != new_state:
#             patients.append(patient)
#             transition_info.append(new_state)
#             state = new_state
#     else:
#         state = new_state
#         run_patient = patient

# patients = np.array(patients)
# transition_info = np.array(transition_info)

# states = list(set(results["state_ids"]))
# paths = (
#     list(product(states, repeat=3))
#     + list(product(states, repeat=4))
#     + list(product(states, repeat=5))
#     + list(product(states, repeat=6))
# )
# path_counts = []
# patient_names = list(set(data["patient"]))

# for patient in patient_names:
#     transition_str = "->".join(transition_info[patients == patient])
#     for path_ in paths:
#         path_str = "->".join(path_)
#         n = transition_str.count(path_str)
#         m = len(path_)
#         if n > 0:
#             path_counts.append(
#                 pd.Series(
#                     {
#                         "count": n,
#                         "transitions": m,
#                         "patient": patient,
#                         "path_sequence": path_str,
#                     }
#                 )
#             )


# path_counts = (
#     pd.concat(path_counts, axis=1)
#     .T.sort_values(
#         by=["transitions", "path_sequence", "count", "patient"], ascending=False
#     )
#     .reset_index(drop=True)
# )
# path_counts.to_csv(common_path / "path_counts.csv")


# %% personalized patient analysis:
# start by the general model, re-fit the model - test performance on patient and get a kappa per patient

subjects = data.patient.unique()
sleep_measures_hypno = {}
sleep_measures_dont_retrain = {}
sleep_measures_retrain = {}


for subject in subjects:

    subject_path = hmm_path / "ica" / "subjects" / subject
    os.makedirs(subject_path, exist_ok=True)
    os.makedirs(subject_path / "retrain", exist_ok=True)
    os.makedirs(subject_path / "dont_rtrain", exist_ok=True)
    subject_data = data.query(f"patient=='{subject}'").reset_index()
    sleep_measures_hypno[subject] = get_sleep_measures_for_patient(
        subject_data[label_col]
    )
    # todo: fold all this into a function and run twice
    #   once with the subject model
    #   once with the global model - in this case only update the transition probabilities
    for retrain in ["dont_rtrain", "retrain"]:

        if retrain == "retrain":
            subject_model, subject_state_to_stage = train_and_map(
                subject_data[ic_names],
                subject_data[label_col],
                lengths=None,
                n_states=best_groups,
                start_from=model_all,
            )
            # drop model to pickle
            with open(subject_path / retrain / "refitted_model.pkl", "wb") as file:
                pickle.dump(subject_model, file)
        else:
            subject_model = model_all
            subject_state_to_stage = state_to_stage

        bic, adjusted_log_likelihood, accuracy, kappa, results = evaluate(
            subject_model,
            subject_state_to_stage,
            subject_data[ic_names],
            labels=subject_data[label_col],
            lengths=None,
            return_samples=True,
        )

        results["time"] = subject_data.time.values
        results.to_csv(subject_path / retrain / "state_assignment_results.csv")
        model_metrics = pd.Series(
            {
                "bic": bic,
                "adjusted_log_likelihood": adjusted_log_likelihood,
                "accuracy": accuracy,
                "kappa": kappa,
            }
        )
        model_metrics.to_csv(subject_path / retrain / "model_metrics.csv")

        state_info = pd.DataFrame(subject_state_to_stage, index=["mapped_states_all"]).T
        state_info["highest_rate"] = results.groupby("hidden_states")["time"].apply(
            lambda x: np.histogram(x, bins=8)[0].argmax()
        )
        state_info["state_ids_all"] = [
            f"{k} ({v})" for k, v in subject_state_to_stage.items()
        ]
        state_info = state_info.reset_index().rename(columns={"index": "hidden_state"})
        state_info = state_info.sort_values(
            ["highest_rate", "mapped_states_all"], ascending=[True, False]
        )

        state_label_counts = plot_hidden_state_stage_distribution(
            subject_model,
            results,
            stage_color_map,
            state_info["state_ids_all"].values,
            save_path=subject_path / retrain / "state_proportion_in_hidden.svg",
            alpha_value=0.7,
            title=None,
        )

        state_counts = (
            pd.DataFrame(state_label_counts)
            .fillna(0)
            .rename(columns=lambda x: int(x[0]), index=lambda x: f"{x}_count")
            .reset_index()
            .rename(columns={"index": "parameter"})
        )

        summary_df = create_state_profile(
            subject_model, results, subject_state_to_stage, ic_names
        )
        summary_df = pd.concat([state_counts, summary_df], axis=0)
        summary_df.to_csv(subject_path / retrain / "model_summary_info.csv")
        subject_hidden_stage_color_map = {
            f"{k} ({v})": PARAMETERS["stage_color_map"][v]
            for k, v in subject_state_to_stage.items()
        }
        if retrain == "retrain":
            plot_parameter_means_and_ci(
                subject_model,
                feature_names=ic_names,
                stage_color_map=subject_hidden_stage_color_map,
                state_order=list(reversed(list(state_info["state_ids_all"].values))),
                save_path=subject_path / retrain / "state_parameter_distribution.svg",
            )

        plot_state_time_histogram(
            results,
            bins=30,
            stage_color_map=stage_color_map,
            state_order=state_info.hidden_state.values,
            save_path=subject_path / retrain / "state_time_distribution.svg",
        )

        # 4. Plot the cross patient state transition probability matrix
        # from the resulting hypnogram estimate the sleep statistics using the yasa function for the human and algo labels
        if retrain == "retrain":
            transition_probs = pd.DataFrame(
                data=subject_model.transmat_,
                columns=range(subject_model.n_components),
                index=range(subject_model.n_components),
            )
            sleep_measures_retrain[subject] = get_sleep_measures_for_patient(
                results.hidden_states_mapped
            )

        else:
            # estimate the transition probabilities from the predicted sequence for the patient
            transition_probs = pd.DataFrame(
                data=estimate_transition_probabilities(results["hidden_states"]),
                columns=range(subject_model.n_components),
                index=range(subject_model.n_components),
            )
            sleep_measures_dont_retrain[subject] = get_sleep_measures_for_patient(
                results.hidden_states_mapped
            )

        plot_transition_graph(
            transition_probs,
            subject_path / retrain / "transition_prob.svg",
            manual=False,
            state_map=subject_state_to_stage,
        )

        # save patient results
        results.to_csv(subject_path / retrain / "results.csv")

        del subject_model


# estimate bland_altman_plot, calculate_icc
sleep_measures_hypno = pd.DataFrame(sleep_measures_hypno).T.fillna(0)
sleep_measures_dont_retrain = pd.DataFrame(sleep_measures_dont_retrain).T.fillna(0)
sleep_measures_retrain = pd.DataFrame(sleep_measures_retrain).T.fillna(0)

sleep_measure_path = hmm_path / "ica" / "sleep_measures"
os.makedirs(sleep_measure_path, exist_ok=True)
sleep_measures_hypno.to_csv("measures_from_hypno.csv")
sleep_measures_dont_retrain.to_csv("measures_from_shared_model.csv")
sleep_measures_retrain.to_csv("measures_from_refitted_model.csv")

compare_sleep_and_unsupervised_measures(
    sleep_measures_all=sleep_measures_hypno,
    unsupervised_measures_all=sleep_measures_dont_retrain,
    save_path=sleep_measure_path / "baseline_to_shared_model",
)

compare_sleep_and_unsupervised_measures(
    sleep_measures_all=sleep_measures_hypno,
    unsupervised_measures_all=sleep_measures_retrain,
    save_path=sleep_measure_path / "baseline_to_retrain_model",
)

compare_sleep_and_unsupervised_measures(
    sleep_measures_all=sleep_measures_dont_retrain,
    unsupervised_measures_all=sleep_measures_retrain,
    save_path=sleep_measure_path / "shared_to_retrain_model",
)

print("whats next")
# individual analysis:
# retrain an individual model and do same evaluation
# how similar is the mapping to the general model?
# estimate sleep statistics - is it better for the individual model?
# analyze the transition probabilities between stages
# 1. is rem stage stability reduced in patient with early onset REM or reduced REM sleep
# 2. is N3 stage stability related to other params?
# 3. is N2 stage stability related to other

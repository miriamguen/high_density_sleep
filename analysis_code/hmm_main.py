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
from joblib import Parallel, delayed

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
    plot_stage_time_histogram,
)

from sleep_evaluation_utils import (
    get_sleep_measures_for_patient,
    compare_sleep_and_unsupervised_measures,
)

# Load parameters
with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)

# output_dir = Path(PARAMETERS["OUTPUT_DIR"])
output_dir = Path(PARAMETERS["OUTPUT_DIR"])
data = pd.read_parquet(output_dir / "processed_data" / "transformed_data.parquet")
label_col = "stage"
pca_ica = "ica"
electrodes = PARAMETERS["select_channels"]
six_all = "all" if electrodes == "None" else "six"
window = PARAMETERS["window_length"]
stage_color_map = PARAMETERS["stage_color_map"]
color_map_name = f"state_color_map_{pca_ica}_{window}_{six_all}"
if color_map_name in PARAMETERS:
    state_color_map = PARAMETERS[color_map_name]
else:
    state_color_map = None

search_range = (2, 16)
n_comp = data.columns.str.startswith("ica").sum()
ic_names = [f"{pca_ica}{i}" for i in range(n_comp)]
alias = f"{pca_ica}_{n_comp}"
hmm_path = output_dir / "hmm"
os.makedirs(hmm_path / alias, exist_ok=True)


def add_time(df):
    df["time"] = df["time_from_onset"] / df["time_from_onset"].max()
    return df


# add time for the analysis of state- time
data = (
    data.groupby("patient")
    .apply(add_time, include_groups=False)
    .reset_index(drop=False)
)

# use the main components, see if there are saved experiment results an only run
search_range = np.arange(search_range[0], search_range[1])
if os.path.exists(hmm_path / alias / f"{alias}_state_scan.csv"):
    results_df = pd.read_csv(hmm_path / alias / f"{alias}_state_scan.csv", index_col=0)
    join_previous_exp = True
    results_df = results_df.sort_values(by=["n_states", "index"])
    search_range = np.array(list(set(search_range) - set(results_df["n_states"])))
else:
    join_previous_exp = False


# scan the state range from 2 to 25 using leave one out cross validation
if len(search_range) > 0:
    results = []

    results = Parallel(n_jobs=PARAMETERS["n_jobs"])(
        delayed(fit_and_score_cv)(data, ic_names, label_col, n_states, scale=False)
        for n_states in tqdm(search_range)
    )
    # # save the cross validation results
    results = pd.concat(results, axis=0)
    results = results.assign(data=f"results_{alias}")

    if join_previous_exp:
        results = pd.concat([results_df, results], axis=0)

    results = results.sort_values(by=["n_states", "index"])
    results.to_csv(hmm_path / alias / f"{alias}_state_scan.csv")
else:
    results = results_df


# use the CV results to select the model using the BICand aic  score, by evaluating a consistent reduction, yet not a label fit,
# to keep selection it unsupervised
best_groups = select_best(results.drop(columns=["kappa"]))
best_cross_val_results = results.loc[results["n_states"] == best_groups]


# plot the evaluation results and the selected state
plot_evaluation_metrics(
    results_df=results,
    best_states=[best_groups],
    save_path=hmm_path / alias / f"{alias}_best_n_states.svg",
)

# %% Shared state analysis:
# 1. Train a model over patients and evaluate overall
common_path = hmm_path / alias / "common_model"
os.makedirs(common_path, exist_ok=True)
lengths = data.groupby("patient").size().tolist()
model_all, state_to_stage, patient_mapping = train_and_map(
    data[ic_names], data[label_col], lengths, best_groups
)

# save the common model
with open(common_path / "model.pkl", "wb") as file:
    pickle.dump(model_all, file)

with open(common_path / "model_mapping.pkl", "wb") as file:
    pickle.dump(state_to_stage, file)

bic, aic, adjusted_log_likelihood, accuracy, kappa, results = evaluate(
    model_all,
    state_to_stage,
    data[ic_names],
    labels=data[label_col],
    lengths=lengths,
    return_samples=True,
)

results["time"] = data.time.values
results["time_from_onset"] = data.time_from_onset.values
results["patient"] = data.patient.values


results.to_csv(common_path / "state_assignment_results.csv")
model_metrics = pd.Series(
    {
        "bic": bic,
        "aic": aic,
        "adjusted_log_likelihood": adjusted_log_likelihood,
        "accuracy": accuracy,
        "kappa": kappa,
    }
)
model_metrics.to_csv(common_path / "model_metrics.csv")
best_cross_val_results.describe().to_csv(common_path / "model_metrics_cv_summary.csv")

state_info = pd.DataFrame(state_to_stage, index=["mapped_states_all"]).T
state_info["highest_rate"] = results.groupby("hidden_states")["time"].apply(
    lambda x: np.histogram(x, bins=3)[0].argmax()
)

state_names = [f"{k} ({v})" for k, v in state_to_stage.items()]
transitions = pd.DataFrame(
    data=model_all.transmat_, index=state_names, columns=state_names
)

state_info["state_ids_all"] = [f"{k} ({v})" for k, v in state_to_stage.items()]
state_info["self_stability"] = [transitions.loc[i, i] for i in state_names]
state_info = state_info.reset_index().rename(columns={"index": "hidden_state"})
state_info = state_info.sort_values(
    ["mapped_states_all", "highest_rate", "self_stability"],
    ascending=[True, False, True],
)

sorted_states = state_info.state_ids_all.values
transitions = transitions.loc[sorted_states, sorted_states]

# Plot transition matrix heatmap
plt.figure(figsize=(10, 8))
# Create mask for low probability transitions
mask = transitions < 0.001

sns.heatmap(
    transitions,
    annot=True,
    fmt=".3f",
    cmap="PuRd",
    mask=mask,
    cbar_kws={"label": "Transition Probability"},
)

plt.title("State Transition Probabilities")
plt.xlabel("To State")
plt.ylabel("From State")
plt.tight_layout()
plt.savefig(common_path / "transition_matrix_next.svg")
plt.close()


G_next = plot_transition_graph(
    transitions,  # .rename(index=lambda x: int(x.split(" ")[0]), columns=lambda x: int(x.split(" ")[0]))
    save_path=common_path / "transition_prob_next.svg",
    state_color_map=state_color_map,
    state_map=state_to_stage,
    th=0.001,
    labels=False,
)
plt.close()

transitions.to_csv(common_path / "transition_matrix_next.csv")

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
    .rename(columns=lambda x: int(x.split(" ")[0]), index=lambda x: f"{x}_count")
    .reset_index()
    .rename(columns={"index": "parameter"})
)
summary_df = create_state_profile(model_all, results, state_to_stage, ic_names)
summary_df = pd.concat([state_counts, summary_df], axis=0)
summary_df.to_csv(common_path / "model_summary_info.csv")

# look at the IC value distribution of each hidden state, of the different labeled sleep stages within each hidden - to asses for similarity
state_order_id = state_info["state_ids_all"].values
state_order_num = state_info["hidden_state"].values
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

    ax.set_yticks(list(range(len(state_order_id))))
    ax.set_yticklabels(
        list(state_order_id),
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
    state_order=list(reversed(list(state_order_id))),
    save_path=common_path / "state_parameter_distribution.svg",
)

plot_state_time_histogram(
    results,
    bins=50,
    stage_color_map=stage_color_map,
    state_order=state_order_num,
    save_path=common_path / "state_time_distribution.svg",
    state_color_map=state_color_map,
)
plt.close()
plot_stage_time_histogram(
    results,
    bins=50,
    stage_color_map=stage_color_map,
    state_order=["W", "N3", "N2", "N1", "R"],
    save_path=common_path / "sleep_stage_time_distribution.svg",
)
plt.close()

# 4. Plot the cross patient state transition probability matrix
G = plot_transition_graph(
    transitions,
    common_path / "transition_prob.svg",
    state_color_map=state_color_map,
    state_map=state_to_stage,
)

transition_probs_mapped = estimate_transition_probabilities(
    results["hidden_states_mapped"]
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
sleep_measures_cross_subject = {}
sleep_measures_shared_model = {}
sleep_measures_personalized = {}

model_metrics = {"cross_subject": {}, "shared_model": {}, "personalized": {}}
print("starting personalized patient analysis")
for subject in subjects:

    subject_path = hmm_path / alias / "subjects" / subject
    os.makedirs(subject_path, exist_ok=True)
    os.makedirs(subject_path / "personalized", exist_ok=True)
    os.makedirs(subject_path / "shared_model", exist_ok=True)
    os.makedirs(subject_path / "cross_subject", exist_ok=True)

    subject_data = data.query(f"patient=='{subject}'").reset_index()
    train_data = data.query(f"patient!='{subject}'").reset_index()
    print(len(train_data))
    sleep_measures_hypno[subject] = get_sleep_measures_for_patient(
        subject_data.stage, sampling_rate=60 / PARAMETERS["window_length"]
    )
    # todo: fold all this into a function and run twice
    #   retrain - the overall model with updated transition probabilities
    #   shared_model - the shared model without changes
    #   cross_patient with a model trained on all other patients
    for retrain in ["shared_model", "cross_subject", "personalized"]:

        if retrain == "personalized":
            subject_model, subject_state_to_stage, _ = train_and_map(
                subject_data[ic_names],
                subject_data[label_col],
                lengths=None,
                n_states=best_groups,
                start_from=model_all,
            )
            # drop model to pickle
            with open(subject_path / retrain / "refitted_model.pkl", "wb") as file:
                pickle.dump(subject_model, file)
        elif retrain == "shared_model":
            subject_model = model_all
            subject_state_to_stage = state_to_stage
        else:
            lengths = train_data.groupby("patient").size().tolist()
            subject_model, subject_state_to_stage, _ = train_and_map(
                train_data[ic_names],
                train_data[label_col],
                lengths=lengths,
                n_states=best_groups,
                start_from=None,
            )

        bic, aic, adjusted_log_likelihood, accuracy, kappa, results = evaluate(
            subject_model,
            subject_state_to_stage,
            subject_data[ic_names],
            labels=subject_data[label_col],
            lengths=None,
            return_samples=True,
        )

        subject_state_to_stage

        results["time"] = subject_data.time.values
        results.to_csv(subject_path / retrain / "state_assignment_results.csv")
        model_metrics[retrain][subject] = {
            "bic": bic,
            "aic": aic,
            "adjusted_log_likelihood": adjusted_log_likelihood,
            "accuracy": accuracy,
            "kappa": kappa,
        }

        state_info = pd.DataFrame(subject_state_to_stage, index=["mapped_states_all"]).T
        state_info["highest_rate"] = results.groupby("hidden_states")["time"].apply(
            lambda x: np.histogram(x, bins=8)[0].argmax()
        )
        state_info["state_ids_all"] = [
            f"{k} ({v})" for k, v in subject_state_to_stage.items()
        ]
        state_info = state_info.reset_index().rename(columns={"index": "hidden_state"})

        if retrain == "personalized":
            state_info = state_info.sort_values(by="highest_rate")
            state_order_id_s = state_info["state_ids_all"].values
            state_order_num_s = state_info["hidden_state"].values
        else:
            state_info = state_info.loc[state_order_num, :]
            state_order_id_s = state_order_id
            state_order_num_s = state_order_num

        state_label_counts = plot_hidden_state_stage_distribution(
            subject_model,
            results,
            stage_color_map,
            state_order_id,
            save_path=subject_path / retrain / "state_proportion_in_hidden.svg",
            alpha_value=0.7,
            title=None,
        )

        state_counts = (
            pd.DataFrame(state_label_counts)
            .fillna(0)
            .rename(
                columns=lambda x: int(x.split(" ")[0]), index=lambda x: f"{x}_count"
            )
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
        if retrain == "personalized":
            state_order_p = list(reversed(list(state_order_id)))
            subject_hidden_stage_color_map = {
                s: PARAMETERS["stage_color_map"][s.split("(")[1].split(")")[0]]
                for s in state_order_p
            }
            plot_parameter_means_and_ci(
                subject_model,
                feature_names=ic_names,
                stage_color_map=subject_hidden_stage_color_map,
                state_order=state_order_p,
                save_path=subject_path / retrain / "state_parameter_distribution.svg",
            )
            sleep_measures_personalized[subject] = get_sleep_measures_for_patient(
                results.hidden_states_mapped,
                sampling_rate=60 / PARAMETERS["window_length"],
            )
        elif retrain == "cross_subject":
            sleep_measures_cross_subject[subject] = get_sleep_measures_for_patient(
                results.hidden_states_mapped,
                sampling_rate=60 / PARAMETERS["window_length"],
            )
        else:
            sleep_measures_shared_model[subject] = get_sleep_measures_for_patient(
                results.hidden_states_mapped,
                sampling_rate=60 / PARAMETERS["window_length"],
            )

        plot_state_time_histogram(
            results,
            bins=30,
            stage_color_map=stage_color_map,
            state_order=state_order_num_s,
            save_path=subject_path / retrain / "state_time_distribution.svg",
        )

        plot_stage_time_histogram(
            results,
            bins=300,
            stage_color_map=stage_color_map,
            state_order=["W", "N3", "N2", "N1", "R"],
            save_path=subject_path / retrain / "sleep_stage_time_distribution.svg",
        )

        # 4. Plot the cross patient state transition probability matrix
        # from the resulting hypnogram estimate the sleep statistics using the yasa function for the human and algo labels
        # estimate the transition probabilities from the predicted sequence for the patient
        transition_probs = estimate_transition_probabilities(results["hidden_states"])

        transition_probs = transition_probs.sort_values(
            by=list(transition_probs.index.values)
        )
        transition_probs = transition_probs.loc[
            transition_probs.index, transition_probs.index
        ]

        plot_transition_graph(
            transition_probs,
            subject_path / retrain / "transition_prob_hidden.svg",
            state_map=subject_state_to_stage,
        )

        transition_probs_mapped = estimate_transition_probabilities(
            results["hidden_states_mapped"]
        )

        plot_transition_graph(
            transition_probs_mapped,
            subject_path / retrain / "transition_prob_mapped.svg",
            state_color_map=stage_color_map,
        )

        if retrain == "shared_model":
            state_order_num_s = list(
                filter(
                    lambda x: int(x) in transition_probs.columns.values,
                    state_order_num_s,
                )
            )
            transition_probs = transition_probs.loc[
                state_order_num_s, state_order_num_s
            ].rename(
                index=lambda x: f"{x} ({subject_state_to_stage[x]})",
                columns=lambda x: f"{x} ({subject_state_to_stage[x]})",
            )
            os.makedirs(common_path / "by_subject", exist_ok=True)
            # Plot transition matrix heatmap
            plt.figure(figsize=(len(state_order_num_s), len(state_order_num_s) * 0.8))
            # Create mask for low probability transitions
            mask = transitions < 0.001

            sns.heatmap(
                transition_probs,
                annot=True,
                fmt=".3f",
                cmap="PuRd",
                mask=mask,
                cbar_kws={"label": "Transition Probability"},
            )

            plt.title("State Transition Probabilities")
            plt.xlabel("To State")
            plt.ylabel("From State")
            plt.tight_layout()
            plt.savefig(
                common_path / "by_subject" / f"{subject}_transition_matrix_next.svg"
            )
            plt.close()

            G_next = plot_transition_graph(
                transition_probs,
                save_path=common_path
                / "by_subject"
                / f"{subject}_transition_prob_next.svg",
                state_color_map=state_color_map,
                state_map=state_to_stage,
                th=0.001,
                labels=False,
            )
            plt.close()

        # save patient results
        results.to_csv(subject_path / retrain / "results.csv")

        del subject_model

sleep_measure_path = hmm_path / alias / "sleep_measures"
os.makedirs(sleep_measure_path, exist_ok=True)
_ = [
    pd.DataFrame(model_metrics[key]).to_csv(
        sleep_measure_path / f"{key}_model_metrics.csv"
    )
    for key in model_metrics.keys()
]

# estimate bland_altman_plot, calculate_icc
sleep_measures_hypno = pd.DataFrame(sleep_measures_hypno).T.fillna(0)
sleep_measures_shared_model = pd.DataFrame(sleep_measures_shared_model).T.fillna(0)
sleep_measures_personalized = pd.DataFrame(sleep_measures_personalized).T.fillna(0)
sleep_measures_cross_subject = pd.DataFrame(sleep_measures_cross_subject).T.fillna(0)

os.makedirs(sleep_measure_path, exist_ok=True)
sleep_measures_hypno.to_csv(sleep_measure_path / "measures_from_hypno.csv")

sleep_measures_shared_model.to_csv(
    sleep_measure_path / "measures_from_shared_model.csv"
)
sleep_measures_personalized.to_csv(
    sleep_measure_path / "measures_from_personalized_model.csv"
)

sleep_measures_cross_subject.to_csv(
    sleep_measure_path / "measures_from_cross_subject_model.csv"
)

compare_sleep_and_unsupervised_measures(
    sleep_measures_all=sleep_measures_hypno,
    unsupervised_measures_all=sleep_measures_cross_subject,
    save_path=sleep_measure_path / "baseline_to_cross_patient_model",
)

compare_sleep_and_unsupervised_measures(
    sleep_measures_all=sleep_measures_hypno,
    unsupervised_measures_all=sleep_measures_personalized,
    save_path=sleep_measure_path / "baseline_to_personalized_model",
)

compare_sleep_and_unsupervised_measures(
    sleep_measures_all=sleep_measures_hypno,
    unsupervised_measures_all=sleep_measures_shared_model,
    save_path=sleep_measure_path / "baseline_to_shared_model",
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

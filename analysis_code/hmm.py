import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, cohen_kappa_score
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)

output_dir = Path(PARAMETERS["OUTPUT_DIR"])
# load the PC transformed data
data = pd.read_parquet(output_dir / "processed_data" / "transformed_data.parquet")
label_col = "stage"
hmm_path = output_dir / "hmm"
statistical_model_path = output_dir / "statistical_models"
os.makedirs(hmm_path, exist_ok=True)
os.makedirs(statistical_model_path, exist_ok=True)
os.makedirs(hmm_path / "ica", exist_ok=True)
os.makedirs(hmm_path / "pca", exist_ok=True)
pc_names_full = data.columns.values[data.columns.str.startswith("pca")]  # ,,
pc_names = ["pca0", "pca1", "pca2", "pca3", "pca4", "pca5"]
ic_names = data.columns.values[
    data.columns.str.startswith("ica")
]  # ["ica0", "ica1", "ica2", "ica3", "ica4", "ica5"]

results_pc = []
results_ic = []
results_pc_ic = []

def scale_patient(data, feature_names):
    df = data.copy(deep=True)
    df[feature_names] = (
        df[feature_names] - np.mean(df[feature_names], axis=0)
    ) / np.std(df[feature_names], axis=0)

    return df


def train_and_map(X, labels, lengths, n_states):
    model = hmm.GaussianHMM(
        n_components=n_states, covariance_type="full", n_iter=100, random_state=42
    )
    model.fit(X, lengths=lengths)

    # Step 3: get the training estimated Hidden States
    hidden_states = model.predict(X, lengths=lengths)

    # Step 4: Evaluate Alignment
    # Map hidden states to the most frequent corresponding stage label
    state_to_stage = {}
    for state in range(n_states):
        most_common_stage = Counter(labels[hidden_states == state]).most_common(1)[0][0]
        state_to_stage[state] = most_common_stage

    return model, state_to_stage


def evaluate(model, state_to_stage, X, labels, lengths=None, return_samples=False):

    # Calculate the log likelihood and number of parameters
    log_likelihood, posteriors = model._score(
        X, lengths=lengths, compute_posteriors=True
    )
    n_params = (n_states**2 + 2 * n_states * X.shape[1] - 1)  # Transition, emission, and initial probabilities
    n_data_points = X.shape[0]

    # Calculate BIC
    bic = n_params * np.log(n_data_points) - 2 * log_likelihood

    hidden_states = model.predict(X, lengths=lengths)
    # Apply mapping to hidden states for evaluation
    hidden_states_mapped = np.vectorize(state_to_stage.get)(hidden_states)

    # Calculate adjusted rand index to measure clustering alignment
    ari_score = adjusted_rand_score(labels, hidden_states_mapped)
    # (Optional) Calculate Accuracy if there is a one-to-one mapping
    accuracy = accuracy_score(labels, hidden_states_mapped)

    kappa = cohen_kappa_score(labels, hidden_states_mapped)

    if return_samples:
        result = pd.DataFrame(
            data=posteriors,
            columns=range(0, len(state_to_stage)),
            index=range(0, len(n_data_points)),
        )
        result = result.assign(
            hidden_states=hidden_states,
            hidden_states_mapped=hidden_states_mapped,
            labels=labels,
        )
        return bic, ari_score, accuracy, kappa, result

    else:
        return bic, ari_score, accuracy, kappa


# search for the optimal state number in PC and ICA:
def fit_and_score(data_raw, feature_names, label_col, n_states, scale=False):
    data = data_raw.copy()
    if scale:
        data = (
            data.groupby("patient")
            .apply(lambda x: scale_patient(x, feature_names), include_groups=False).reset_index()
        )

    all_patient_results = {}
    # Leave-One-Out Cross-Validation by patient
    unique_patients = data["patient"].unique()
    for patient in unique_patients:
        # Split into train and test based on patient
        train_df = data[data["patient"] != patient]
        test_df = data[data["patient"] == patient]
        # train the model specifying the sequence lengths for training data
        train_lengths = train_df.groupby("patient").size().tolist()
        train_labels = train_df[label_col].values
        train_data = train_df[feature_names].values
        model, state_to_stage = train_and_map(
            X=train_data, labels=train_labels, n_states=n_states, lengths=train_lengths
        )
        # evaluate on the test patient
        test_labels = test_df[label_col].values
        test_data = test_df[feature_names].values
        bic, ari_score, accuracy, kappa = evaluate(
            model,
            state_to_stage,
            X=test_data,
            labels=test_labels,
            lengths=None,
            return_samples=False,
        )
        all_patient_results[patient] = {
            "bic": bic,
            "ari_score": ari_score,
            "accuracy": accuracy,
            "kappa": kappa,
            "n_states": n_states,
            "state_to_stage": str(state_to_stage),
        }

    return pd.DataFrame(all_patient_results).T.reset_index()


def plot_evaluation_metrics(
    results_df: pd.DataFrame,
    savepath: str,
    best_state: list,
    metrics: list = ["bic", "kappa", "accuracy"],
) -> None:
    """
    Plots BIC, ARI, Accuracy, and Kappa metrics for different states.

    Args:
        results_df (pd.DataFrame): DataFrame with evaluation metrics for each state.
        savepath:
    Returns:
        None
    """
    x = results_df["n_states"].unique().astype(int) - 4
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(int(np.ceil(len(x) / 3 + 2)), 3 * len(metrics)),
        sharex="col",
    )
    axes = axes.ravel()
    for i, metric in enumerate(metrics):
        sns.boxplot(
            x="n_states",
            y=metric,
            data=results_df,
            ax=axes[i],
            color="lightblue",
            fliersize=0,
        )
        sns.scatterplot(
            x=results_df["n_states"] - 4,
            y=metric,
            data=results_df,
            color="darkblue",
            s=15,
            ax=axes[i],
            alpha=0.3,
        )

        state_means = results_df.groupby("n_states")[metric].mean()
        state_medians = results_df.groupby("n_states")[metric].median()

        # Add mean and median trend lines
        axes[i].plot(x, state_means.values, color="gray", linestyle="--", label="Mean")
        axes[i].plot(
            x, state_medians.values, color="black", linestyle="-.", label="Median"
        )

        if metric != "bic":
            axes[i].set_ylim([0, 1])
            axes[i].set_ylabel(metric.capitalize(), fontsize=14)
        else:
            axes[i].legend()
            axes[i].set_ylabel(metric.upper(), fontsize=14)

        # Highlight best state
        optimal_hidden_states = min(best_state)
        best_state_max = results_df[results_df["n_states"] == optimal_hidden_states][
            metric
        ].max()
        best_state_min = results_df[results_df["n_states"] == optimal_hidden_states][
            metric
        ].min()

        axes[i].axhline(
            best_state_max,
            color="forestgreen",
            linestyle=":",
            label=f"Best State: {best_state}",
        )
        axes[i].axhline(
            best_state_min,
            color="forestgreen",
            linestyle=":",
            label=f"Best State: {best_state}",
        )
        for state in best_state:
            max_val = results_df[results_df["n_states"] == state][metric].max()
            min_val = results_df[results_df["n_states"] == state][metric].min()
            if metric != "bic":
                axes[i].text(
                    state - 4,
                    min_val * 0.90,
                    f"{min_val:.2f}-{max_val:.2f}",
                    color="forestgreen",
                    ha="center",
                    va="top",
                    fontsize=10,
                    rotation=90,
                )

    axes[i].set_xlabel("Hidden States", fontsize=14)
    axes[i].set_xticks(x)

    plt.tight_layout(pad=1)
    plt.savefig(savepath)
    plt.close()


# find
for n_states in tqdm(range(4, 40)):
    results_pc_ic.append(
        fit_and_score(data, pc_names + ic_names, label_col, n_states, scale=False)
    )
    results_pc.append(fit_and_score(data, pc_names, label_col, n_states, scale=False))
    results_ic.append(fit_and_score(data, ic_names, label_col, n_states, scale=False))



results_ic = pd.concat(results_ic, axis=0)
results_ic = results_ic.assign(data="results_ic")
results_ic.to_csv(hmm_path / "ica_n_state_scan.csv")

results_pc = pd.concat(results_pc, axis=0)
results_pc = results_pc.assign(data="results_pc")
results_pc.to_csv(hmm_path / "pca_n_state_scan.csv")


for n_states in tqdm(range(4, 7)):
    


th = 0.75
# select the optimal state based on kappa:
pc_kappa_grouped = results_pc.groupby("n_states")["kappa"]
condition_1 = pc_kappa_grouped.min() > pc_kappa_grouped.min().quantile(th)
condition_2 = pc_kappa_grouped.max() > pc_kappa_grouped.max().quantile(th)
groups = np.logical_and(condition_1, condition_2)
best_groups = groups[groups].index.values


# we can see that 4, 37 hidden states (in the 30 second segment) give a comparable cross subject performance range


plot_evaluation_metrics(
    results_df=results_pc,
    best_state=best_groups,  # .query("n_states in @best_groups"),
    savepath=hmm_path / "pca_best_n_states.svg",
)


# select the optimal state based on kappa:
ic_kappa_grouped = results_ic.groupby("n_states")["kappa"]
condition_1 = ic_kappa_grouped.min() > ic_kappa_grouped.min().quantile(th)
condition_2 = ic_kappa_grouped.max() > ic_kappa_grouped.max().quantile(th)
groups = np.logical_and(condition_1, condition_2)
best_groups = groups[groups].index.values


# we can see that 15, 25 hidden states (in the 30 second segment) give a comparable cross subject performance range
plot_evaluation_metrics(
    results_df=results_ic,
    best_state=best_groups,  # .query("n_states in @best_groups"),
    savepath=hmm_path / "ica_best_n_states.svg",
)

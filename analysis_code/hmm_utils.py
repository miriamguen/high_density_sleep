# hmm_utils.py
import os
from typing import Tuple, Dict, Any, List
from collections import Counter

import numpy as np
import pandas as pd

from hmmlearn import hmm
from sklearn.metrics import accuracy_score, adjusted_rand_score, cohen_kappa_score

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns


def scale_patient(data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Scales features for each patient to have zero mean and unit variance.

    Args:
        data: Data for scaling.
        feature_names: List of feature columns to scale.

    Returns:
        Scaled data as a DataFrame.
    """
    df = data.copy(deep=True)
    df[feature_names] = (
        df[feature_names] - np.mean(df[feature_names], axis=0)
    ) / np.std(df[feature_names], axis=0)
    return df


def train_and_map(
    X: np.ndarray,
    labels: np.ndarray,
    lengths: List[int],
    n_states: int,
    start_from: hmm.GaussianHMM = None,
) -> Tuple[hmm.GaussianHMM, Dict[int, int]]:
    """Trains HMM and maps hidden states to observed labels.

    Args:
        X: Training data array.
        labels: True labels for the training data.
        lengths: Sequence lengths for training data.
        n_states: Number of hidden states for HMM.
        start_from: optional, use to start from the parameter state of a previously fitted models
    Returns:
        Trained HMM model and state-to-label mapping.
    """

    if start_from is None:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            tol=1.0e-2,
            random_state=42,
        )
    else:
        # Retrieve parameters from the original model
        startprob = start_from.startprob_.copy()
        transmat = start_from.transmat_.copy()
        means = start_from.means_.copy()
        covars = start_from.covars_.copy()

        # Create a new model and set the retrieved parameters
        model = hmm.GaussianHMM(
            n_components=start_from.n_components,
            covariance_type="full",
            params="stmc",  # update the transition probabilities
            init_params="",
            n_iter=10,
            tol=1.0e-1,
            random_state=42,
        )
        # Set init_params="" to prevent re-initialization

        # # Assign parameters to the new model
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars

    model = model.fit(X, lengths=lengths)
    _, posteriors = model._score(X, lengths=lengths, compute_posteriors=True)

    state_to_stage = {}
    patient_mapping = {}
    # use the sample with the maximal posterior probability for each training patient to set the state- making this minimally supervised
    # This approach will allow the labelers to only define one label per state in each patient, prompt by the system
    for state in range(n_states):
        if lengths is not None:
            i = 0
            best_label = []
            for length in lengths:
                best_label.append(
                    labels[i + np.argmax(posteriors[i : i + length, state])]
                )
                i = i + length
            state_to_stage[state] = max(best_label, key=best_label.count)
            patient_mapping[state] = best_label
        else:
            state_to_stage[state] = labels[np.argmax(posteriors[:, state])]
            patient_mapping[state] = state_to_stage[state]

    return model, state_to_stage, patient_mapping


def careful_predict(model, X, lengths, emission_gap=0.8):
    """
    This function uses careful predictions, in cases were the emission probability is high this overrides the
    Viterbi Algorithm predictions, this may allow some cases of sharper transitions where the data shows the shift

    important - in shorter segments this may not be recommended, or the emission_th should be increased
    """
    log_likelihood, posteriors = model._score(
        X, lengths=lengths, compute_posteriors=True
    )
    emission_states = np.argmax(posteriors, axis=1)
    emission_diff = -np.min(posteriors.T - np.max(posteriors, axis=1), axis=0)
    hidden_states = model.predict(X, lengths=lengths)
    cond_1 = hidden_states != emission_states

    if np.sum(cond_1) > 0:
        cond_2 = emission_diff > emission_gap
        override = np.logical_and(cond_1, cond_2)
        hidden_states[override] = emission_states[override]

    return log_likelihood, posteriors, hidden_states


def evaluate(
    model: hmm.GaussianHMM,
    state_to_stage: Dict[int, int],
    X: np.ndarray,
    labels: np.ndarray,
    lengths: List[int] = None,
    return_samples: bool = False,
) -> Tuple[float, float, float, float, Any]:
    """Evaluates HMM performance on test data.

    Args:
        model: Trained HMM model.
        state_to_stage: Mapping from hidden states to labels.
        X: Test data.
        labels: True labels for test data.
        lengths: Sequence lengths for the test data.
        return_samples: If True, returns sample-level results.

    Returns:
        BIC, ARI, accuracy, kappa, and (optionally) posteriors.
    """
    log_likelihood, posteriors, hidden_states = careful_predict(
        model, X, lengths=lengths, emission_gap=0.8
    )

    n_data_points = X.shape[0]

    bic = model.bic(X)
    aic = model.aic(X)
    adjusted_log_likelihood = log_likelihood / (
        len(X) * np.sqrt(np.var(X, axis=0).sum())
    )

    hidden_states_mapped = np.vectorize(state_to_stage.get)(hidden_states)
    accuracy = accuracy_score(labels, hidden_states_mapped)
    kappa = cohen_kappa_score(labels, hidden_states_mapped)

    if return_samples:
        result = pd.DataFrame(
            data=posteriors,
            columns=range(0, len(state_to_stage)),
            index=range(0, n_data_points),
        )
        result = result.assign(
            hidden_states=hidden_states,
            hidden_states_mapped=hidden_states_mapped,
            labels=labels,
        )
        return bic, aic, adjusted_log_likelihood, accuracy, kappa, result
    else:
        return bic, aic, adjusted_log_likelihood, accuracy, kappa


def fit_and_score_cv(
    data: pd.DataFrame,
    feature_names: list,
    label_col: str,
    n_states: int,
    cv_col: str = "patient",
    scale: bool = False,
    return_model: bool = False,
) -> pd.DataFrame:
    """Fits HMM and scores results for each patient using leave-one-out CV.

    Args:
        data: Raw data to fit and score.
        feature_names: List of features for HMM.
        label_col: Column with stage labels.
        cv_col: typical the patient / subject indicator column to cross validate over
        n_states: Number of hidden states.
        scale: Whether to scale data by patient.
        return_model: bool, wether to return a dictionary with all models and stage maps.
    Returns:
        DataFrame with scores for each patient.
    """
    if scale:
        data = (
            data.groupby(cv_col)
            .apply(lambda x: scale_patient(x, feature_names), include_groups=False)
            .reset_index()
        )

    all_patient_results = {}
    if return_model:
        models = {}
    # Leave-One-Out Cross-Validation by patient
    unique_patients = data[cv_col].unique()
    for patient in unique_patients:
        # Split into train and test based on patient
        train_df = data[data[cv_col] != patient]
        test_df = data[data[cv_col] == patient]
        # train the model specifying the sequence lengths for training data
        train_lengths = train_df.groupby(cv_col).size().tolist()
        train_labels = train_df[label_col].values
        train_data = train_df[feature_names].values
        model, state_to_stage, patient_mapping = train_and_map(
            X=train_data, labels=train_labels, n_states=n_states, lengths=train_lengths
        )
        # evaluate on the test patient
        test_labels = test_df[label_col].values
        test_data = test_df[feature_names].values
        bic, aic, adjusted_log_likelihood, accuracy, kappa = evaluate(
            model,
            state_to_stage,
            X=test_data,
            labels=test_labels,
            lengths=None,
            return_samples=False,
        )

        all_patient_results[patient] = {
            "bic": bic,
            "aic": aic,
            "adjusted_log_likelihood": adjusted_log_likelihood,
            "accuracy": accuracy,
            "kappa": kappa,
            "n_states": n_states,
            "state_to_stage": str(state_to_stage),
        }
    if return_model:
        models[patient] = {
            "model": model,
            "state_to_stage": state_to_stage,
            "patient_mapping": patient_mapping,
        }
        return models, pd.DataFrame(all_patient_results).T.reset_index()
    else:
        return pd.DataFrame(all_patient_results).T.reset_index()


def select_best(results: pd.DataFrame) -> np.ndarray:
    """
    Selects optimal hidden states based on the balance between model likelihood and complexity
    If a kappa value is passed in the DataFrame the selection of a higher state requires an increase in minimal agreement.

    Args:
        results: DataFrame containing evaluation metrics, including 'kappa' and 'n_states'.

    Returns:
        np.ndarray: Array of state indices that satisfy both minimum and maximum kappa conditions.
    """
    # Allow added complexity only if bic is reduced
    bic = results.groupby("n_states")["bic"].median()
    improved = [True]
    false_count = 0
    l = bic.values[0]
    for m_l in bic.values[1:]:
        if m_l < l:
            l = m_l
            improved.append(True)
            false_count = 0
        else:
            improved.append(False)
            false_count += 1

    groups_b = np.where(improved)[0]
    bic_improve = bic.iloc[list(groups_b)]
    candidates_b = bic_improve[
        (bic_improve.diff() / bic_improve) < -0.0001
    ]  # state increase induces at least 0.1% improvement

    aic = results.groupby("n_states")["aic"].median()
    improved = [True]
    false_count = 0
    l = aic.values[0]

    for m_l in aic.values[1:]:
        if m_l < l:
            l = m_l
            improved.append(True)
            false_count = 0
        else:
            improved.append(False)
            false_count += 1

    groups_a = np.where(improved)[0]
    aic_improve = aic.iloc[list(groups_a)]
    candidates_a = aic_improve[(aic_improve.diff() / aic_improve) < -0.0001]

    candidates = list(
        set(candidates_a.index.values).intersection(candidates_b.index.values)
    )

    return max(candidates)


def plot_evaluation_metrics(
    results_df: pd.DataFrame,
    best_states: List[int],
    save_path: str,
    metrics: list = ["bic", "aic", "kappa", "accuracy"],
) -> None:
    """Plots evaluation metrics for different numbers of hidden states.

    Args:
        results_df: DataFrame with evaluation metrics.
        best_states: List of states to highlight on the plot.
        savepath: Path to save the resulting plot.
        metrics: the metrics to plot
    Returns:
        None
    """
    lowest_hidden = np.min(results_df["n_states"].values)
    x = results_df["n_states"].unique().astype(int) - lowest_hidden
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(int(np.ceil(len(x) / 3 + 2)), 3 * len(metrics)),
        sharex="col",
    )
    axes = axes.ravel()
    optimal_hidden_states = min(best_states)
    for i, metric in enumerate(metrics):
        sns.violinplot(
            x="n_states",
            y=metric,
            data=results_df,
            ax=axes[i],
            color="lightgray",
            # fliersize=0,
            inner="quart",
            density_norm="count",
            cut=0,
        )

        state_medians = results_df.groupby("n_states")[metric].median()

        # Add mean and median trend lines
        axes[i].plot(
            x,
            state_medians.values,
            color="black",
            label="Median",
            lw=1,
            linestyle="--",
            alpha=0.7,
        )

        if not (metric in ["bic", "aic"]):
            if metric == "adjusted_log_likelihood":
                axes[i].set_ylabel(metric.capitalize().replace("_", " "), fontsize=14)
            else:
                axes[i].set_ylim([0, 1])
                axes[i].set_ylabel(metric.capitalize(), fontsize=14)
            # Highlight best state
            max_val = results_df[results_df["n_states"] == optimal_hidden_states][
                metric
            ].max()
            min_val = results_df[results_df["n_states"] == optimal_hidden_states][
                metric
            ].min()

            axes[i].axhline(
                max_val,
                color="forestgreen",
                linestyle=":",
                label=f"{metric.capitalize()} range for {optimal_hidden_states} states",
            )
            axes[i].axhline(
                min_val,
                color="forestgreen",
                linestyle=":",
            )
            low, high = axes[i].get_ylim()
            y_range = 0.05 * (high - low)

            axes[i].text(
                optimal_hidden_states - lowest_hidden,
                max_val + y_range,
                f"{max_val:.2f}",
                color="forestgreen",
                ha="center",
                va="center",
                fontsize=10,
                rotation=0,
            )

            axes[i].text(
                optimal_hidden_states - lowest_hidden,
                min_val - y_range,
                f"{min_val:.2f}",
                color="forestgreen",
                ha="center",
                va="center",
                fontsize=10,
                rotation=0,
            )

        else:
            max_val = results_df[metric].max()
            min_val = results_df[metric].min()
            mean_val = (max_val + min_val) / 2
            range_val = max(mean_val - min_val, max_val - mean_val) * 1.3
            axes[i].set_ylim([mean_val - range_val, mean_val + range_val])
            axes[i].set_ylabel(metric.upper(), fontsize=14)
            best_state_median = results_df[
                results_df["n_states"] == optimal_hidden_states
            ][metric].median()

            axes[i].axhline(
                best_state_median,
                color="forestgreen",
                linestyle=":",
                label=f"Best fit: {optimal_hidden_states} states",
            )

        axes[i].legend(loc="lower right")

    axes[i].set_xlabel("Hidden States", fontsize=14)
    axes[i].set_xticks(x)
    plt.tight_layout(pad=1)
    plt.savefig(save_path)
    plt.close()


def plot_hidden_state_stage_distribution(
    model: hmm.GaussianHMM,
    results: pd.DataFrame,
    stage_color_map: Dict[str, str],
    state_order: list,
    save_path: str,
    alpha_value: float = 0.7,
    title: str = None,
) -> None:
    """
    Plots pie charts showing label proportions within each unique hidden state.

    Args:
        results: DataFrame containing 'hidden_states_mapped' and 'labels' columns.
        stage_color_map: Dictionary mapping sleep stage labels to colors.
        save_path: File path to save the plot.
        alpha_value: Alpha transparency for pie chart colors.

    Returns:
        None
    """
    # Get unique states and apply alpha to stage colors
    unique_states = results["hidden_states"].unique()
    state_prevalence = model.get_stationary_distribution()
    colors_with_alpha = {
        label: to_rgba(color, alpha=alpha_value)
        for label, color in stage_color_map.items()
    }
    # Set up the plot layout with one subplot per hidden state
    fig, axes = plt.subplots(
        len(unique_states), 1, figsize=(4.5, len(unique_states) * 4.5)
    )

    state_order_ = list(filter( lambda x: int(x.split(" ")[0]) in unique_states, state_order))

    state_label_counts = {}
    for i, state in enumerate(state_order_):
        if int(state.split(" ")[0]) not in results["hidden_states"]:
            continue

        state_data = results[results["hidden_states"] == int(state.split(" ")[0])][
            "labels"
        ]
        state_name = results[results["hidden_states"] == int(state.split(" ")[0])][
            "hidden_states_mapped"
        ].unique()

        state_name
        if len(state_name) != 1:
            continue

        state_name = state_name[0]
        label_counts = state_data.value_counts()
        state_label_counts[state] = label_counts.to_dict()
        # Get colors for each label based on color mapping
        colors = [colors_with_alpha[label] for label in label_counts.index]

        wedges, texts, autotexts = axes[i].pie(
            label_counts,
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
            startangle=0,
            colors=colors,
            pctdistance=1.17,
        )

        # Set font size for percentage labels
        for autotext in autotexts:
            autotext.set_fontsize(14)

        state_num = int(state.split(" ")[0])
        axes[i].set_xlabel(
            f"{state}: {100*state_prevalence[state_num]:.0f}%",
            fontsize=22,
        )

        if i == len(unique_states) - 1:
            # Add legend on the last subplot
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=to_rgba(color, alpha=alpha_value),
                    linestyle="",
                    markersize=14,
                )
                for color in stage_color_map.values()
            ]
            axes[i].legend(
                handles,
                stage_color_map.keys(),
                title="Sleep Stages",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )

    if title != None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return state_label_counts


def estimate_transition_probabilities(sequence: np.ndarray) -> pd.DataFrame:
    """
    Estimates the transition probability matrix for a sequence of sleep stages.

    Args:
        sequence (np.ndarray): Sequence of observed sleep stages (e.g., hypnogram).

    Returns:
        pd.DataFrame: Transition probability matrix as a DataFrame.
    """
    unique_states = np.unique(sequence)
    n_states = len(unique_states)

    # Initialize a matrix to count transitions
    transition_counts = np.zeros((n_states, n_states), dtype=int)
    state_to_index = {state: i for i, state in enumerate(unique_states)}

    # Count transitions
    for current_state, next_state in zip(sequence[:-1], sequence[1:]):
        i, j = state_to_index[current_state], state_to_index[next_state]
        transition_counts[i, j] += 1

    # Calculate probabilities by normalizing counts
    transition_probabilities = transition_counts / transition_counts.sum(
        axis=1, keepdims=True
    )

    return pd.DataFrame(
        transition_probabilities, index=unique_states, columns=unique_states
    )


def estimate_median_sequence_length(sequence: np.ndarray) -> pd.Series:
    """
    Estimates the mean expected sequence length for each state in the sequence.

    Args:
        sequence (np.ndarray): Sequence of observed sleep stages.

    Returns:
        pd.Series: Mean sequence length for each state.
    """
    # Find lengths of consecutive sequences for each state
    lengths = []
    current_state = sequence[0]
    length = 1
    for state in sequence[1:]:
        if state == current_state:
            length += 1
        else:
            lengths.append((current_state, length))
            current_state = state
            length = 1

    lengths.append((current_state, length))  # Append the final state sequence

    # Convert lengths to a DataFrame and calculate mean length per state
    lengths_df = pd.DataFrame(lengths, columns=["State", "Length"])
    median_lengths = lengths_df.groupby("State")["Length"].median()
    mean_lengths = lengths_df.groupby("State")["Length"].mean()
    number_of_episodes = lengths_df.groupby("State")["Length"].count()
    return median_lengths, mean_lengths, number_of_episodes


def create_circle_coordinates(values):
    """
    Generates (X, Y) coordinates for points spaced equally around a circle.

    Args:
        values (list or array): A vector of values to place on the circle.

    Returns:
        list of tuples: Each tuple contains the (X, Y) coordinates for a point.
    """
    angle_increment = 2 * np.pi / len(values)  # Angle between each point in radians

    coordinates = {}
    for i, state in enumerate(values):
        angle = (np.pi / 2) + i * angle_increment
        x = 1 + np.cos(angle)  # X coordinate
        y = 1 + np.sin(angle)  # Y coordinate
        coordinates[state] = (x, y)

    return coordinates


def plot_transition_graph(
    transition_probs_: pd.DataFrame, save_path: str, manual=True, state_map=None
) -> None:
    """
    Plots a directed graph for the transition probabilities with custom layout.

    Args:
        transition_probs_ (pd.DataFrame): Transition probability matrix as a DataFrame.
        save_path: str
        manual: if manual labels or not
        state_map: if hidden stats are passed and not manual provide the state mapping
    Returns:
        None
    """
    transition_probs = transition_probs_.copy()
    # Custom layout for states
    stage_color_map = {
        "W": "orange",
        "R": "red",
        "N1": "magenta",
        "N2": "royalblue",
        "N3": "midnightblue",
    }

    states = transition_probs.index.values
    pos = create_circle_coordinates(states)

    if not (manual):
        # n_hidden = len(state_map)
        # if n_hidden == 9:
        #     pos = {
        #         0: (2.5, 8),
        #         5: (5.5, 8),
        #         3: (0, 5.5),
        #         1: (8, 5.5),
        #         4: (4, 4),
        #         2: (0, 2.5),
        #         6: (8, 2.5),
        #         8: (2, 0),
        #         7: (6, 0),
        #     }
        # else:
        #     states = transition_probs.index.values
        #     pos = create_circle_coordinates(states)

        hidden_states = list(pos.keys())
        renaming = lambda n: f"{n} ({state_map[n]})"
        pos = {renaming(k): v for k, v in pos.items()}
        states = [renaming(n) for n in hidden_states]
        stage_color_map = {
            f"{n} ({state_map[n]})": stage_color_map[state_map[n]]
            for n in hidden_states
        }
        transition_probs = transition_probs.rename(index=renaming, columns=renaming)

    transition_probs = transition_probs.map(lambda x: 0 if x < 0.01 else np.round(x, 2))
    # estimate in and out node degree
    transition_node = transition_probs.map(lambda x: 1 if x >= 0.01 else 0)
    out_degree = transition_node.sum(axis=1)
    in_degree = transition_node.sum(axis=0)
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and weighted edges based on non-zero transition probabilities
    self_loops = {}  # Dictionary to store self-transition probabilities for each state

    # Add nodes and weighted edges based on non-zero transition probabilities
    for i, state_from in enumerate(transition_probs.index):
        for j, state_to in enumerate(transition_probs.columns):
            prob = transition_probs.iloc[i, j]
            if prob > 0:
                if state_from == state_to:
                    # Store self-transition probability for later display inside the node
                    self_loops[state_from] = prob
                else:
                    # Add only non-self-transition edges
                    G.add_edge(
                        state_from,
                        state_to,
                        weight=prob,
                        color=stage_color_map[state_from],
                    )

    plt.figure(figsize=(8, 8))
    # Draw nodes
    # Draw nodes with colors from the stage_color_map
    node_colors = {node: stage_color_map[node] for node in pos.keys()}
    node_sizes = 4000 * np.emath.logn(len(in_degree), in_degree.loc[pos.keys()]) + 1000
    node_alpha = 1 - 0.9 * (out_degree / len(out_degree)).loc[pos.keys()]

    nx.draw_networkx_nodes(
        G,
        nodelist=list(pos.keys()),
        pos=pos,
        node_size=list(node_sizes),
        node_color=list(node_colors.values()),
        edgecolors="black",
        alpha=list(node_alpha.values),
    )

    # Draw edges with colors and widths based on transition probability
    edges = G.edges(data=True)
    edge_colors = [
        edge[2]["color"] for edge in edges
    ]  # Get color based on the origin node
    edge_widths = [
        edge[2]["weight"] * 10 for edge in edges
    ]  # Scale width for visibility
    # Draw labels for edges (non-self probabilities)
    edge_labels = {(edge[0], edge[1]): f"{edge[2]['weight']:.2f}" for edge in edges}
    edge_label_pos = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=10,
        label_pos=0.25,
    )
    # Adjust edge label colors to match arrow colors
    for (n1, n2), label in edge_label_pos.items():
        label.set_color(stage_color_map[n1])

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=edge_widths,
        edge_color=edge_colors,
        arrowstyle="->",
        arrowsize=25,
        connectionstyle="arc3,rad=0.2",
        min_source_margin=35,
        min_target_margin=35,
    )

    # Draw labels for nodes
    nx.draw_networkx_labels(
        G, pos, font_size=12, font_color="black", verticalalignment="bottom"
    )

    # Draw self-transition labels inside nodes
    for node, prob in self_loops.items():
        x, y = pos[node]
        plt.text(
            x,
            y - 0.09,
            f"{prob:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            # fontweight="bold",
        )

    # Display plot

    plt.title("Sleep Stage Transition Probabilities")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()
    return G


def plot_parameter_means_and_ci(
    model: hmm.GaussianHMM,
    stage_color_map: Dict[str, str],
    feature_names: list,
    state_order: list,
    save_path: str,
):
    """
    Plots the mean and 95% confidence interval of each parameter distribution for each state,
    specifically for a GHMM with 'full' covariance matrices.

    Args:
        model (hmm.GaussianHMM): The fitted Gaussian Hidden Markov Model.
        feature_names (list of str, optional): List of feature names for subplot titles.
    """
    n_states = model.n_components
    n_features = model.means_.shape[1]
    states = [int(state.split(" ")[0]) for state in state_order]
    color_order = [stage_color_map[x] for x in state_order]
    # Set up subplots for each feature
    fig, axes = plt.subplots(
        1, n_features, figsize=(3 * n_features, n_states), sharey="row"
    )
    fig.suptitle("Parameter Distribution", fontsize=16)

    # Ensure `axes` is iterable even if there's only one feature
    if n_features == 1:
        axes = [axes]

    for feature_idx in range(n_features):
        means = model.means_[states, feature_idx]

        # Calculate standard deviations from the diagonal of the full covariance matrix
        std_devs = np.sqrt(
            [model.covars_[state][feature_idx, feature_idx] for state in states]
        )

        # Scatter plot for means with error bars tor the parameter STD
        axes[feature_idx].scatter(
            means,
            range(n_states),
            color=color_order,  # "royalblue",
            edgecolor="black",
            s=100,
            zorder=3,
        )
        axes[feature_idx].errorbar(
            means,
            range(n_states),
            xerr=std_devs,
            fmt="o",
            color="black",  # color_order,#"royalblue",
            capsize=5,
            zorder=2,
            lw=2,
        )

        # Set subplot title and labels
        if feature_idx == 0:
            axes[feature_idx].set_ylabel("State", fontsize=13)
            axes[feature_idx].set_yticks(list(range(n_states)))
            axes[feature_idx].set_yticklabels(
                list(state_order),
                rotation=90,
                va="center",
                fontsize=12,
            )

        parts = feature_names[feature_idx].split("a")
        axes[feature_idx].set_xlabel(
            f"{parts[0].upper()} {int(parts[1])+1}", fontsize=13
        )

        if feature_names[feature_idx].upper() != "TIME":
            x_lim = (
                np.max([np.max(means + std_devs), np.max(np.abs(means - std_devs))])
                * 1.1,
            )
            round_lim = np.round(x_lim, -1)
            if round_lim > x_lim:
                x_lim = round_lim
            else:
                x_lim = round_lim + 10

            axes[feature_idx].set_xlim([-x_lim, x_lim])
            ticks = int(x_lim * 0.5)
            axes[feature_idx].set_xticks([-ticks, 0, ticks])
            axes[feature_idx].set_xticklabels(
                axes[feature_idx].get_xticklabels(), ha="center", fontsize=12
            )
        else:
            axes[feature_idx].set_xlim([0, 1])
            axes[feature_idx].set_xticks([0.1, 0.9])
            axes[feature_idx].set_xticklabels(
                ["Start", "End"], ha="center", fontsize=12
            )

    plt.tight_layout(pad=1)  # rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_state_time_histogram(results, bins, stage_color_map, state_order, save_path):
    """
    Plots the time distribution for each hidden state as a histogram with a KDE overlay.

    Args:
        results (pd.DataFrame): DataFrame containing 'hidden_state' and 'time' columns.
        bins (int): Number of bins for the histogram.
        stage_color_map:
        state_order: the order in which to plot the stats
        save_path: use directory and file name joined
    """
    if state_order is None:
        hidden_states = results["hidden_states"].unique()
    else:
        hidden_states = state_order

    # Set up the figure for subplots
    n_states = len(hidden_states)
    fig, axes = plt.subplots(n_states, 1, figsize=(5, 1.5 * n_states), sharex=True)
    # fig.suptitle("Time Distribution by Hidden State", fontsize=16)

    # If there's only one hidden state, make `axes` iterable
    if n_states == 1:
        axes = [axes]

    # Plot histogram with KDE overlay for each hidden state
    for i, state in enumerate(hidden_states):
        state_data = results[results["hidden_states"] == state]
        if len(state_data) == 0:
            continue
        mapped_state = state_data["hidden_states_mapped"].values[0]
        color = stage_color_map[mapped_state]
        # Plot histogram and KDE
        sns.histplot(
            state_data["time"],
            bins=bins,
            binrange=(0, 1),
            kde=True,
            ax=axes[i],
            color=color,  # "royalblue",
            edgecolor="black",
        )
        axes[i].set_xlim(0, 1)  # Set range for histogram
        axes[i].set_ylabel(f"{state} ({mapped_state}) count")

    # Label the x-axis of the last subplot
    axes[-1].set_xlabel("Relative Time")

    plt.tight_layout(pad=1)
    plt.savefig(save_path)
    plt.close()


def plot_stage_time_histogram(results, bins, stage_color_map, state_order, save_path):
    """
    Plots the time distribution for each hidden state as a histogram with a KDE overlay.

    Args:
        results (pd.DataFrame): DataFrame containing 'hidden_state' and 'time' columns.
        bins (int): Number of bins for the histogram.
        stage_color_map:
        state_order: the order in which to plot the stats
        save_path: use directory and file name joined
    """
    if state_order is None:
        states = results["labels"].unique()
    else:
        states = state_order

    # Set up the figure for subplots
    n_states = len(states)
    fig, axes = plt.subplots(n_states, 1, figsize=(5, 1.5 * n_states), sharex=True)
    # fig.suptitle("Time Distribution by Hidden State", fontsize=16)

    # If there's only one hidden state, make `axes` iterable
    if n_states == 1:
        axes = [axes]

    # Plot histogram with KDE overlay for each hidden state
    for i, state in enumerate(states):
        state_data = results[results["labels"] == state]
        if len(state_data) == 0:
            continue

        color = stage_color_map[state]
        # Plot histogram and KDE
        sns.histplot(
            state_data["time"],
            bins=bins,
            binrange=(0, 1),
            kde=True,
            ax=axes[i],
            color=color,  # "royalblue",
            edgecolor="black",
        )
        axes[i].set_xlim(0, 1)  # Set range for histogram
        axes[i].set_ylabel(f"{state} count")

    # Label the x-axis of the last subplot
    axes[-1].set_xlabel("Relative Time")

    plt.tight_layout(pad=1)
    plt.savefig(save_path)
    plt.close()


def create_state_profile(model, results, state_to_stage, feature_names):
    """
    Generates a summary table to profile the states of a Gaussian Hidden Markov Model (GHMM)
    for model comparison across patients. This table includes transition probabilities,
    node degrees, self-transition probabilities, state mappings, feature means, feature
    standard deviations, and time distribution histograms.

    Args:
        model (hmmlearn.hmm.GaussianHMM): The trained GHMM model to analyze.
        results (pd.DataFrame): DataFrame containing 'hidden_states' and 'time' columns.
        state_to_stage (dict): Mapping of state IDs to labels (e.g., sleep stages).
        feature_names: the feature_names used in the model
    Returns:
        pd.DataFrame: A summary DataFrame profiling each state and including transition
                      information, mean and standard deviation of features, and time
                      histograms for comparison between patients.
    """
    # Get the list of unique states from the model
    states = list(state_to_stage.keys())
    prevelance = model.get_stationary_distribution()
    state_prevlence = pd.Series(
        {state: prevelance[state] for state in states}, name="state_prevlence"
    )
    # Create transition probability matrix as a DataFrame
    state_profile = pd.DataFrame(data=model.transmat_, columns=states, index=states)

    # Calculate the out-degree and in-degree for each node, thresholding by a minimum probability (0.01)
    out_node_degree = (state_profile > 0.01).sum(axis=1).rename("out_node_degree")
    in_node_degree = (state_profile > 0.01).sum(axis=0).rename("in_node_degree")

    # Calculate self-transition probability for each state
    self_probability = pd.Series(
        {state: np.sqrt(model.transmat_[state, state]) for state in states},
        name="self_probability",
    )

    # Map states to their corresponding labels
    state_map = pd.Series(state_to_stage, name="state_map")

    # Rename the rows of the transition matrix for clarity and convert to a DataFrame
    state_profile = (
        state_profile.rename(index=lambda x: f"from {x}")
        .reset_index()
        .rename(columns={"index": "parameter"})
    )

    # Calculate the mean of each feature for each state
    feature_mean = pd.DataFrame(
        data=model.means_.T, columns=states, index=feature_names
    )
    feature_mean = (
        feature_mean.rename(index=lambda x: f"mean {x}")
        .reset_index()
        .rename(columns={"index": "parameter"})
    )

    # Calculate standard deviation for each feature in each state
    n_features = model.n_features
    std_devs = {}
    for feature_idx in range(n_features):
        # Extract standard deviations from the diagonal of each state's covariance matrix
        std_devs[feature_idx] = {
            state: np.sqrt(model.covars_[state][feature_idx, feature_idx])
            for state in states
        }

    # Convert standard deviations to a DataFrame
    feature_stds = (
        pd.DataFrame(std_devs)
        .T.rename(index=lambda x: feature_names[x])
        .reset_index()
        .rename(columns={"index": "parameter"})
    )

    # Create a histogram of time values for each hidden state
    time_histogram = (
        results.groupby("hidden_states")["time"]
        .apply(lambda x: str(list(np.histogram(x, 30)[0])))
        .rename("time_histogram")
    )

    # Combine the summary statistics into a single DataFrame
    summary_df = (
        pd.concat(
            [
                state_map,
                self_probability,
                in_node_degree,
                out_node_degree,
                time_histogram,
            ],
            axis=1,
        )
        .T.reset_index()
        .rename(columns={"index": "parameter"})
    )

    # Add state profiles, feature means, and standard deviations to the summary
    summary_df = pd.concat(
        [summary_df, state_profile, feature_mean, feature_stds], axis=0
    )

    return summary_df

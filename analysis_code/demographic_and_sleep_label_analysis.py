"""Demographics and Sleep Measures Analysis:
The following code integrates demographic exploration and sleep measures distribution,
including violin plots for key sleep parameters"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv("Details information for healthy subjects.csv")

# Plot settings from YAML file
AGE_BINS = 7
GENDER_COLORS = {"M": "orange", "F": "cornflowerblue"}

# Initialize figure
fig, ax = plt.subplots(3, 3, figsize=(21, 15))

# Pie chart for gender distribution
df.groupby("Sex").size().plot(
    kind="pie",
    autopct=lambda val: f"{val / 100 * len(df):.0f}\n{val:.0f}%",
    textprops={"fontsize": 16},
    colors=[GENDER_COLORS["F"], GENDER_COLORS["M"]],
    ax=ax[0, 0],
)

# Histograms and
plot_columns = [
    ("Age", (1, 0)),
    ("SOL(min)", (0, 1)),
    ("SE(%TRT)", (1, 1)),
]
for x, (i, j) in plot_columns:
    sns.histplot(
        data=df,
        x=x,
        hue="Sex",
        kde=True,
        stat="proportion",
        common_norm=False,
        ax=ax[i, j],
        legend=True,
        bins=AGE_BINS,
    )

# scatter plots for sleep measures
plot_columns = [
    ("SOL(min)", "WASO(min)", (0, 2)),
    ("TST(min)", "SE(%TRT)", (1, 2)),
    ("TST(min)", "N2(min)", (2, 2)),
    ("TST(min)", "N3(min)", (2, 1)),
    ("TST(min)", "R(min)", (2, 0)),
]
for x, y, (i, j) in plot_columns:
    sns.scatterplot(data=df, x=x, y=y, hue="Sex", size="Age", ax=ax[i, j])

# Sleep measures violin plots using Plotly
df = df.rename(columns=lambda x: x.replace("(min)", ""))

sleep_measures = {
    "TRT": ("lightblue", "mediumblue"),
    "TST": ("lightpink", "deeppink"),
    "SOL": ("mediumpurple", "purple"),
    "REML": ("lightgray", "gray"),
    "WASO": ("yellowgreen", "green"),
    "N1": ("lightseagreen", "black"),
    "N2": ("lightgreen", "green"),
    "N3": ("rosybrown", "Red"),
    "R": ("yellowgreen", "green"),
}

violins = []

for param, color in sleep_measures.items():
    violins.append(
        go.Violin(
            y=df[param],
            box_visible=True,
            line_color=color[1],
            meanline_visible=True,
            fillcolor=color[0],
            opacity=0.6,
            points="all",
            name=param,
        )
    )

# Create the violin plot figure
fig_violin = go.Figure(data=violins)
fig_violin.update_layout(
    title="Sleep Macrostructure of Healthy Subjects",
    xaxis_title="Sleep Parameters",
    yaxis_title="Durations (min)",
    font_size=16,
    plot_bgcolor="white",
    width=1000,
    margin=dict(t=100, pad=10),
    legend=dict(font_size=14),
)
import os

# Save the figures
os.makedirs("output/figures", exist_ok=True)
fig.tight_layout(pad=1)
fig.savefig("output/figures/demographic_analysis.svg")
pio.write_image(
    fig_violin, file="output/figures/sleep_measure_distribution.svg", engine="orca"
)
fig.show()
plt.close()


# estimate state stability for subject
from hmm_utils import (
    estimate_transition_probabilities,
    estimate_median_sequence_length,
    plot_transition_graph,
)
from preprocessing_utils import read_hypno_and_plot

results = []
os.makedirs(f"output/transition", exist_ok=True)
os.makedirs(f"output/transition/from_hypno", exist_ok=True)
for subject in df["Subjects ID"]:

    hypno = read_hypno_and_plot(f"{subject}/{subject}/{subject}.txt")
    hypno = hypno.loc[hypno["stage"] != "L", "stage"]
    stage_order = ["W", "N1", "N2", "N3", "R"]
    transition_probs = estimate_transition_probabilities(hypno.values).loc[
        stage_order, stage_order
    ]

    plot_transition_graph(
        transition_probs, f"output/transition/from_hypno/{subject}_transition_prob.svg"
    )
    median_duration, mean_duration, number_of_episodes = (
        estimate_median_sequence_length(hypno.values)
    )
    transition_probs["median_duration"] = median_duration.values
    transition_probs["mean_duration"] = mean_duration.values
    transition_probs["number_of_episodes"] = number_of_episodes.values
    transition_probs["subject"] = subject
    results.append(transition_probs)

result_df = pd.concat(results)

transition_probs_mean = (
    result_df.reset_index().drop(columns=["subject"]).groupby("index").mean()
)
plot_transition_graph(
    transition_probs_mean.loc[stage_order, stage_order],
    f"output/transition/from_hypno/mean_transition_prob.svg",
)


result_df.to_csv("output/transition/transition_prob_time_by_subject.csv")

transition_probs_mean.to_csv("output/transition/transition_prob_time_mean.csv")

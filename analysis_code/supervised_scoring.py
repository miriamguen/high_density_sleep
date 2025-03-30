# This script is used to score the supervised model
# in this script we will use the PCA AND ICA components
# 1. to train a linear svm model
# 2. do so in a cross-subject manner (leave one subject out cross validation)
# 3. score the model using agreement for each state, presenting the confusion matrix

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut

os.getcwd()

with open("analysis_code/parameters.yaml", "r") as f:
    PARAMETERS = yaml.safe_load(f)


data = pd.read_parquet(
    Path(PARAMETERS["OUTPUT_DIR"]) / "decomposition" / "models" / "all_pc_data.parquet"
)

save_dir = Path(PARAMETERS["OUTPUT_DIR"]) / "supervised_scoring"
save_dir.mkdir(parents=True, exist_ok=True)
y = "stage"
stages = ["W", "R", "N1", "N2", "N3"]

over_all_metrics = {}
max_comp = 63

for i in tqdm(range(1, max_comp)):
    train_on = data.columns[data.columns.str.startswith("pca")][0:i]

    for subject in tqdm(data["patient"].unique()):
        subject_data = data[data["patient"] == subject]
        train_data = data[data["patient"] != subject]

        X_train = train_data[train_on]
        y_train = train_data[y]

        X_test = subject_data[train_on]
        y_test = subject_data[y]

        model = LinearSVC(class_weight="balanced")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        run_name = f"{i}_{subject}"

        result = {
            f"precision_{stage}": x
            for stage, x in zip(
                stages, precision_score(y_test, y_pred, labels=stages, average=None)
            )
        }
        result.update(
            {
                f"recall_{stage}": x
                for stage, x in zip(
                    stages, recall_score(y_test, y_pred, labels=stages, average=None)
                )
            }
        )
        result.update(
            {
                f"f1_{stage}": x
                for stage, x in zip(
                    stages, f1_score(y_test, y_pred, labels=stages, average=None)
                )
            }
        )
        result["kappa"] = cohen_kappa_score(y_test, y_pred)
        result["accuracy"] = accuracy_score(y_test, y_pred)
        result["components"] = i

        over_all_metrics[run_name] = result


metrics_df = pd.DataFrame(over_all_metrics).T
metrics_df.to_csv(save_dir / "metrics_df.csv", index=False)


import seaborn as sns


fig, ax = plt.subplots(1, 1, figsize=(20, 5))
sns.lineplot(data=metrics_df, x="components", y="kappa", ax=ax, errorbar=("ci", 95))
ax.set_xlabel("Number of Components")
ax.set_xticks(range(1, max_comp))
ax.set_xticklabels(range(1, max_comp))
ax.set_ylabel("Kappa")

fig.savefig(save_dir / "kappa_comp_num.svg")


fig, ax = plt.subplots(1, 1, figsize=(20, 5))
sns.lineplot(data=metrics_df, x="components", y="accuracy", ax=ax, errorbar=("ci", 95))
ax.set_xlabel("Number of Components")
ax.set_xticks(range(1, max_comp))
ax.set_xticklabels(range(1, max_comp))
ax.set_ylabel("Accuracy")

fig.savefig(save_dir / "accuracy_comp_num.svg")

stage_colors = PARAMETERS["stage_color_map"]

# precision by stage
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
for stage in stages:
    sns.lineplot(
        data=metrics_df,
        x="components",
        y=f"precision_{stage}",
        ax=ax,
        errorbar=("ci", 95),
        color=stage_colors[stage],
    )
ax.set_xlabel("Number of Components")
ax.set_xticks(range(1, max_comp))
ax.set_xticklabels(range(1, max_comp))
ax.set_ylabel("Precision")
fig.savefig(save_dir / "precision_comp_num.svg")


# recall by stage
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
for stage in stages:
    sns.lineplot(
        data=metrics_df,
        x="components",
        y=f"recall_{stage}",
        ax=ax,
        errorbar=("ci", 95),
        color=stage_colors[stage],
    )
ax.set_xlabel("Number of Components")
ax.set_xticks(range(1, max_comp))
ax.set_xticklabels(range(1, max_comp))
ax.set_ylabel("Recall")
fig.savefig(save_dir / "recall_comp_num.svg")


# f1 by stage
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
for stage in stages:
    sns.lineplot(
        data=metrics_df,
        x="components",
        y=f"f1_{stage}",
        ax=ax,
        errorbar=("ci", 95),
        color=stage_colors[stage],
    )
ax.set_xlabel("Number of Components")
ax.set_xticks(range(1, max_comp))
ax.set_xticklabels(range(1, max_comp))
ax.set_ylabel("F1")
fig.savefig(save_dir / "f1_comp_num.svg")

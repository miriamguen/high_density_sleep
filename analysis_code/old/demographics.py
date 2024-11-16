# -*- coding: utf-8 -*-
"""
create a plot exploring the patient metadata:
1. More males than females
2. females had older age
2.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def label_function(val):
    return f"{val / 100 * len(df):.0f}\n{val:.0f}%"


# plot the age distribution
df = pd.read_csv("Details information for healthy subjects.csv")

## histogram of age by gender
gender = [df["Sex"] == "F", df["Sex"] == "M"]

fig, ax = plt.subplots(3, 3, figsize=(21, 15))

df.groupby("Sex").size().plot(
    kind="pie",
    autopct=label_function,
    textprops={"fontsize": 16},
    colors=["cornflowerblue", "orange"],
    ax=ax[0, 0],
)

# Histograms for each gender
sns.histplot(
    data=df,
    x="Age",
    hue="Sex",
    kde=True,
    stat="proportion",
    common_norm=False,
    ax=ax[0, 1],
    legend=True,
    bins=7,
)


sns.histplot(
    data=df,
    x="SE (% TRT)",
    hue="Sex",
    kde=True,
    stat="proportion",
    common_norm=False,
    ax=ax[1, 1],
    legend=True,
    bins=7,
)
sns.histplot(
    data=df,
    x="SOL (min)",
    hue="Sex",
    kde=True,
    stat="proportion",
    common_norm=False,
    ax=ax[1, 0],
    legend=True,
    bins=7,
)

sns.scatterplot(
    data=df,
    y="SE (% TRT)",
    size="Age",
    x="TST (min)",
    hue="Sex",
    ax=ax[2, 1],
    legend=True,
)
sns.scatterplot(
    data=df,
    x="SOL (min)",
    size="Age",
    y="WASO(min)",
    hue="Sex",
    ax=ax[2, 0],
    legend=True,
)

sns.scatterplot(
    data=df,
    y="N2 (min)",
    size="Age",
    hue="Sex",
    x="TST (min)",
    ax=ax[0, 2],
    legend=True,
)
sns.scatterplot(
    data=df, y="N3(min)", size="Age", hue="Sex", x="TST (min)", ax=ax[1, 2], legend=True
)
sns.scatterplot(
    data=df, y="R(min)", size="Age", hue="Sex", x="TST (min)", ax=ax[2, 2], legend=True
)

# Add title
ax[0, 1].set_title("Histogram of age, by sex", fontsize=16)
ax[0, 0].set_title("Gender", fontsize=16)
ax[1, 0].set_title("Sleep efficiency by gender", fontsize=16)
ax[1, 1].set_title("Sleep latency by gender", fontsize=16)

ax[2, 1].set_title("Sleep efficiency by total sleep time", fontsize=16)
ax[2, 0].set_title("WASO by sleep latency", fontsize=16)

ax[0, 2].set_title("N2 by total sleep time", fontsize=16)
ax[1, 2].set_title("N3 by total sleep time", fontsize=16)
ax[2, 2].set_title("REM by total sleep time", fontsize=16)


# Force legend to appear
# plt.legend()
fig.tight_layout(pad=1)
fig.savefig("demographic.svg")

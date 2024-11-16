import os
import yaml
from pathlib import Path
import warnings
import itertools

from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from decomposition_utils import rename_pc


def plot_coefficients_with_ci(
    data: pd.DataFrame, stage_color_map: dict, ic_name: str
) -> tuple:
    """
    Plots the coefficients with confidence intervals and significance levels for each sleep stage.

    Parameters:
    - data: pd.DataFrame
        DataFrame containing the model results including keys 'Stage', 'Coefficient', 'P>|z|',
        'ci_low', 'ci_high', and patient-specific random effects.
    - stage_color_map: dict
        A dictionary mapping each stage (e.g., 'W', 'R', 'N1', etc.) to its corresponding color.
    - ic_name: str
        The name of the independent component (IC) or principal component (PC) for which the
        coefficients are being plotted.

    Returns:
    - fig: plt.Figure
        The Matplotlib figure object.
    - ax: plt.Axes
        The Matplotlib Axes object for further customization.
    """

    def significance_stars(p_value: float) -> str:
        """
        Returns the significance stars based on the p-value.

        Parameters:
        - p_value: float
            The p-value from the statistical test.

        Returns:
        - str: Significance stars corresponding to the p-value threshold.
        """
        if p_value > 0.001:
            return ""
        elif p_value > 0.0001:
            return "*"
        elif p_value > 0.00001:
            return "**"
        else:
            return "***"

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(5, 5))

    # Identify the columns that represent patient-specific random slopes
    random_cols = list(filter(lambda x: x.startswith("random"), data.columns.values))

    # Calculate patient-specific slopes by adding random effects to the fixed effects
    patient_slopes = data[random_cols] + data[["Coefficient"]].values

    # Determine y-axis limits based on the maximum of confidence intervals and patient-specific slopes
    y_lim = (
        max(
            data[["ci_high", "ci_low"]].abs().max().max(),
            patient_slopes.abs().max().max(),
        )
        * 1.1
    )

    # Set the index for both dataframes
    data = data.set_index("Stage")
    patient_slopes = patient_slopes.set_index(data.index)

    # Plot the coefficient and confidence interval for each stage
    for i, stage in enumerate(stage_color_map.keys()):
        coef = data["Coefficient"][stage]
        ci_lower = data["ci_low"][stage]
        ci_upper = data["ci_high"][stage]
        p_val = data["P>|z|"][stage]
        color = stage_color_map[stage]
        patients_slope = patient_slopes.loc[stage, :]

        # Add vertical confidence interval bars with thicker lines
        ax.plot([i, i], [ci_lower, ci_upper], color=color, alpha=0.7, lw=10)
        ax.scatter(i, coef, color=color, alpha=0.7, s=150)
        ax.scatter(
            i * np.ones(len(patients_slope)), patients_slope, color="grey", alpha=1, s=5
        )
        ax.axhline(
            y=0, color="gray", linestyle="--", label="0", alpha=0.5, linewidth=0.5
        )

        # Add significance stars above the coefficients
        ax.text(
            i,
            max(ci_upper, max(patients_slope)) + 0.01,
            significance_stars(p_val),
            fontsize=12,
            ha="center",
        )

    # Customize the x-axis labels and y-axis
    ax.set_xticks(np.arange(len(stage_color_map.keys())))
    ax.set_xticklabels(stage_color_map.keys(), fontsize=12)
    ax.set_ylabel("Coefficient", fontsize=12)
    ax.set_ylim([-y_lim, y_lim])
    ax.set_title(f"{rename_pc(ic_name)} Sleep Stage Direction", fontsize=14)

    # Add a legend if there are significant p-values
    if data["P>|z|"].min() < 0.001:
        legend_elements = [
            plt.Line2D([0], [0], marker=None, color="w", label="*   p < 0.001"),
            plt.Line2D([0], [0], marker=None, color="w", label="**  p < 0.0001"),
            plt.Line2D([0], [0], color="w", label="*** p < 0.00001"),
        ]
        ax.legend(handles=legend_elements, title="", loc="best")

    plt.tight_layout()
    return fig, ax


# Load the parameters from the configuration file
with open("analysis_code/parameters.yaml", "r") as file:
    PARAMETERS = yaml.safe_load(file)

# Create the output directory for statistical models if it doesn't exist
output_dir = Path(PARAMETERS["OUTPUT_DIR"])
statistical_model_path = output_dir / "statistical_models"
os.makedirs(statistical_model_path, exist_ok=True)

# Load the transformed principal component (PC) data
data = pd.read_parquet(output_dir / "processed_data" / "transformed_data.parquet")

# Define the component names and sleep stages
pc_names = ["pca0", "pca1", "pca2", "pca3", "pca4", "pca5"]
ic_names = ["ica0", "ica1", "ica2", "ica3", "ica4", "ica5"]
stages = ["W", "R", "N1", "N2", "N3"]

# Convert 'stage' and 'patient' columns to categorical data type
data["stage"] = data["stage"].astype("category")
data["patient"] = data["patient"].astype("category")

# Standardize the independent components and principal components for comparable inference coefficients
data[ic_names + pc_names] = data[ic_names + pc_names] / data[ic_names + pc_names].std()


def corr_pairs(data, names):
    pairs = list(itertools.combinations(names, 2))
    correlation = data.corr()
    result = {}
    for a, b in pairs:
        result[f"{a}-{b}"] = correlation.loc[a, b]
    return result


ic_corr_results = {}
pc_corr_results = {}

patients = list(set(data["patient"].values))
for patient in patients:
    patient_data = data.query(f"patient=='{patient}'")
    ic_corr_results[patient] = corr_pairs(patient_data[ic_names], ic_names)
    pc_corr_results[patient] = corr_pairs(patient_data[pc_names], pc_names)

pd.DataFrame(ic_corr_results).T.to_csv(statistical_model_path / "ics_patient_corr.csv")
pd.DataFrame(pc_corr_results).T.to_csv(statistical_model_path / "pcs_patient_corr.csv")

# List to store results from each model
model_results = []

# Fit a mixed effects model for each IC and PC component
for ic in ic_names + pc_names:
    model_result = smf.mixedlm(
        f"{ic} ~ 0 + stage",
        data=data,
        groups=data["patient"],
        re_formula="~0 + stage",
    ).fit()

    # Extract fixed effect coefficients and confidence intervals
    fixed_effects = model_result.fe_params.rename("Coefficient")
    standard_errors = model_result.bse_fe.rename("Std.Err.")
    p_values = model_result.pvalues[fixed_effects.index].rename("P>|z|")
    conf_int = (
        model_result.conf_int()
        .loc[fixed_effects.index, :]
        .rename(columns={0: "ci_low", 1: "ci_high"})
    )

    # Extract random effects for each patient
    random_effects = pd.concat(
        [re_params for group, re_params in model_result.random_effects.items()], axis=1
    ).rename(columns=lambda x: f"random_{x}")

    # Combine fixed and random effects into a DataFrame
    model_data = pd.concat(
        [fixed_effects, standard_errors, p_values, conf_int, random_effects], axis=1
    )
    model_data = model_data.assign(IC=ic)
    model_data = model_data.reset_index(drop=False).rename(columns={"index": "Stage"})
    model_data["Stage"] = (
        model_data["Stage"].str.replace("stage[", "").str.replace("]", "")
    )
    model_data = model_data.assign(converged=model_result.converged)
    model_results.append(model_data)

# Concatenate all model results into a single DataFrame
model_results_df = pd.concat(model_results, axis=0)

# Save the results to a CSV file
model_results_df.to_csv(
    statistical_model_path / "mixed_effects_results.csv", index=False
)

# Plot and save coefficient plots for each component
for ic in ic_names + pc_names:
    data = model_results_df[model_results_df["IC"] == ic]
    fig, ax = plot_coefficients_with_ci(data, PARAMETERS["stage_color_map"], ic)
    fig.savefig(statistical_model_path / f"modeled_{ic}_stage_distribution.svg")
    plt.close()


# plot the naive IC value distribution in each stage
data = pd.read_parquet(output_dir / "processed_data" / "transformed_data.parquet")
for ic in ic_names + pc_names:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(y=0, color="gray", linestyle="--", label="0", alpha=0.75, linewidth=1)
    sns.violinplot(
        data=data,
        y=ic,
        x="stage",
        palette=PARAMETERS["stage_color_map"],
        density_norm="count",
        cut=0,
        ax=ax,
        inner="quart",
        order=PARAMETERS["stage_color_map"].keys(),
    )

    max_val = max(-data[ic].min(), data[ic].max()) * 1.1
    ax.set_ylim(-max_val, max_val)
    fig.savefig(statistical_model_path / f"empirical_{ic}_stage_distribution.svg")
    plt.close()

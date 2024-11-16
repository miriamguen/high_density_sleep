# -*- coding: utf-8 -*-
"""
Anphy
"""
import os
import yasa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path


def read_hypno_and_plot(
    path: Path, plot: bool = False, figure_save_path: Path = None
) -> pd.DataFrame:
    """_summary_

    Args:
        path (Path): The file path
        plot (bool, optional): wether to plot the hypnogram. Defaults to False.
        figure_save_path (Path, optional): if plot is True the path in which to save the plot, if empty will be saved in the original data folder. Defaults to None.

    Returns:
        pd.DataFrame: a table with the hypnogram
    """
    hypno = pd.read_csv(
        path,
        delimiter="\t",
        header=None,
        names=["stage", "time_from_onset", "epoch_length"],
    )

    mapper = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "L": -1}

    hypno["num"] = hypno["stage"].apply(lambda x: mapper[x])

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5), constrained_layout=True)
        ax = yasa.plot_hypnogram(hypno["num"], fill_color="lightblue", ax=ax)
        if figure_save_path == None:
            fig.savefig(str(path).replace(".txt", "hypno.svg"))
        else:
            fig.savefig(figure_save_path)

    return hypno


if __name__ == "__main__":

    base_path = os.getcwd()
    txt_files = filter(
        lambda x: x.startswith("E"), glob.glob("**/*.txt", recursive=True)
    )

    for name in txt_files:
        df = read_hypno_and_plot(Path(name))


print("done")

import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from cycler import cycler
from mle_toolbox import visualize


def sparse(iteration, sparsity=0.8):
    return (sparsity ** iteration) * 100.0


def imp_plot(
    results_dict,
    fig=None,
    ax=None,
    plot_title: str = "IMP Plot",
    xlabel: str = "Weights Remaining [%]",
    ylabel: str = "Performance",
    curve_labels: Union[list, None] = [
        "mask/weights",
        "mask/permuted",
        "permuted/permuted",
        "random/reinit",
    ],
    pruning_ratio: float = 0.2,
    legend_loc: int = 0,
    plot_ylabel: bool = True,
    plot_xlabel: bool = True,
    plot_legend: bool = True,
    num_imp_iters: int = 30,
    colors: list = ["r", "g", "b", "yellow", "k"],
    linestyles=10 * ["-"],
):
    """Classic IMP plot: x-Sparsity, y-Performance."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # default_cycler = cycler(color=colors)

    plt.rc("lines", linewidth=3)
    # plt.rc("axes", prop_cycle=default_cycler)

    num_imp = results_dict[list(results_dict.keys())[0]]["mean"].shape[0]
    num_imp_iters = min(num_imp, num_imp_iters)

    if curve_labels is None:
        i = 0
        for k, v in results_dict.items():
            ax.plot(v["mean"][:num_imp_iters], label=k, c=colors[i])
            ax.fill_between(
                np.arange(v["mean"][:num_imp_iters].shape[0]),
                v["mean"][:num_imp_iters]
                - 1 / np.sqrt(5) * v["std"][:num_imp_iters],
                v["mean"][:num_imp_iters]
                + 1 / np.sqrt(5) * v["std"][:num_imp_iters],
                alpha=0.3,
                color=colors[i],
            )
            i += 1
    else:
        for i, k in enumerate(results_dict):

            ax.plot(
                np.arange(results_dict[k]["mean"][:num_imp_iters].shape[0]),
                results_dict[k]["mean"][:num_imp_iters],
                label=curve_labels[i],
                c=colors[i],
                ls=linestyles[i],
            )
            ax.fill_between(
                np.arange(results_dict[k]["mean"][:num_imp_iters].shape[0]),
                results_dict[k]["mean"][:num_imp_iters]
                - 1 / np.sqrt(5) * results_dict[k]["std"][:num_imp_iters],
                results_dict[k]["mean"][:num_imp_iters]
                + 1 / np.sqrt(5) * results_dict[k]["std"][:num_imp_iters],
                alpha=0.3,
                color=colors[i],
            )

    # Prettify the plot
    if plot_legend:
        ax.legend(fontsize=18, loc=legend_loc)
    ax.set_title(plot_title)
    if plot_ylabel:
        ax.set_ylabel(ylabel)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xtick_labels = [
        round(sparse(i, 1 - pruning_ratio), 2)
        for i in np.arange(0, num_imp_iters, 5)
    ]
    xtick_labels.append(round(sparse(num_imp_iters - 1, 1 - pruning_ratio), 2))
    xticks = np.arange(0, num_imp_iters, 5).tolist()
    xticks.append(num_imp_iters - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(axis="both", which="major", labelsize=20)

    print(xtick_labels)

    fig.tight_layout()
    return fig, ax


def weight_plot(
    weights_es,
    weights_sgd,
    modules_to_consider=["Conv_0/kernel", "Conv_1/kernel", "Dense_0/kernel"],
    title="MNIST: CNN",
):
    vars_to_consider = ["sparsity", "mean_magnitude", "mean"]
    labels = ["Sparsity", r"Mean $|W|$", r"Mean $W$"]
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    counter = 0
    red = (239 / 255, 89 / 255, 151 / 255)
    green = (155 / 255, 242 / 255, 224 / 255)
    for module in modules_to_consider:
        for i, var in enumerate(vars_to_consider):
            summary_imp_es, summary_imp_sgd = [], []
            for imp_iter in weights_es.keys():
                summary_imp_es.append(weights_es[imp_iter][module][var])
                summary_imp_sgd.append(weights_sgd[imp_iter][module][var])

            axs.flatten()[counter].plot(summary_imp_es, label="ES", c=red)
            axs.flatten()[counter].plot(summary_imp_sgd, label="SGD", c=green)
            axs.flatten()[counter].set_title(f"{module}")
            axs.flatten()[counter].spines["top"].set_visible(False)
            axs.flatten()[counter].spines["right"].set_visible(False)
            axs.flatten()[counter].set_ylabel(labels[i])
            if counter > 5:
                axs.flatten()[counter].set_xlabel("IMP Iteration")
            counter += 1
    axs.flatten()[counter - 1].legend()
    fig.suptitle(f"{title} - Layerwise Sparsity & Weight Distribution", y=0.95)
    fig.tight_layout()
    return fig, axs

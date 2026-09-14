import numpy as np
import matplotlib.pyplot as plt

from src.modele import generate_own_model, skewed_group_shares, compute_group_sizes
from src.metryki import (
    gini_coefficient,
    modularity_ground_truth,
    community_detection_scores,
)

N = 1000
K = 5
M0 = 3
M = 3
BETA = 0.1
SKEW_VALUES = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
NUM_RUNS = 100
SEED_BASE = 0


def run_sweep():
    rows = []
    for skew in SKEW_VALUES:
        s = skewed_group_shares(K, skew)
        sizes = compute_group_sizes(N, s)
        gini = gini_coefficient(sizes)

        q_list, nmi_l_list, ari_l_list, nmi_ld_list, ari_ld_list = [], [], [], [], []
        for i in range(NUM_RUNS):
            g = generate_own_model(
                N=N, K=K, m0=M0, m=M, s=s, beta=BETA, seed=SEED_BASE + i
            )
            q_list.append(modularity_ground_truth(g))
            comm = community_detection_scores(g)
            nmi_l_list.append(comm["nmi_louvain"])
            ari_l_list.append(comm["ari_louvain"])
            nmi_ld_list.append(comm["nmi_leiden"])
            ari_ld_list.append(comm["ari_leiden"])

        rows.append(
            {
                "skew": skew,
                "gini": gini,
                "min_size": min(sizes),
                "max_size": max(sizes),
                "Q_mean": np.mean(q_list),
                "Q_std": np.std(q_list),
                "nmi_louvain_mean": np.mean(nmi_l_list),
                "nmi_louvain_std": np.std(nmi_l_list),
                "ari_louvain_mean": np.mean(ari_l_list),
                "ari_louvain_std": np.std(ari_l_list),
                "nmi_leiden_mean": np.mean(nmi_ld_list),
                "nmi_leiden_std": np.std(nmi_ld_list),
                "ari_leiden_mean": np.mean(ari_ld_list),
                "ari_leiden_std": np.std(ari_ld_list),
            }
        )
        print(
            f"skew={skew:<4} Gini={gini:.3f}  rozmiary=[{min(sizes)}..{max(sizes)}]  "
            f"Q={np.mean(q_list):.3f}  NMI(Leiden)={np.mean(nmi_ld_list):.3f}"
        )

    return rows


def plot_detectability_vs_gini(rows, out_path_png):
    gini = [r["gini"] for r in rows]
    q = [r["Q_mean"] for r in rows]
    nmi_l = [r["nmi_louvain_mean"] for r in rows]
    nmi_l_std = [r["nmi_louvain_std"] for r in rows]
    nmi_ld = [r["nmi_leiden_mean"] for r in rows]
    nmi_ld_std = [r["nmi_leiden_std"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        gini, nmi_l, yerr=nmi_l_std, marker="o", capsize=3, label="NMI (Louvain)"
    )
    plt.errorbar(
        gini, nmi_ld, yerr=nmi_ld_std, marker="s", capsize=3, label="NMI (Leiden)"
    )
    plt.plot(
        gini,
        q,
        marker="^",
        linestyle="--",
        color="#2ca02c",
        label="Modularność względem podziału rzeczywistego Q",
    )
    plt.xlabel("Współczynnik Giniego rozmiarów grup (0 = równe, →1 = bardzo nierówne)")
    plt.ylabel("Wartość metryki")
    plt.title(f"Wpływ nierówności rozmiarów grup na wykrywalność (β={BETA})")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print(
        f"Sweep Gini: N={N}, K={K}, m={M}, beta={BETA}, {NUM_RUNS} przebiegow na punkt...\n"
    )
    rows = run_sweep()
    plot_detectability_vs_gini(rows, "outputs/wykrywalnosc_vs_gini.png")
    print("\nZapisano: wykrywalnosc_vs_gini.png")

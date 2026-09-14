import numpy as np
import matplotlib.pyplot as plt

from src.modele import generate_own_model
from src.metryki import clustering, modularity_ground_truth, community_detection_scores

N = 1000
K = 5
M0 = 3
M = 3
S = [0.2] * K
BETA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
NUM_RUNS = 100
SEED_BASE = 0


def run_sweep():
    rows = []
    for beta in BETA_VALUES:
        c_global_list, c_local_list, q_list = [], [], []
        nmi_l_list, ari_l_list, nmi_ld_list, ari_ld_list = [], [], [], []

        for i in range(NUM_RUNS):
            g = generate_own_model(
                N=N, K=K, m0=M0, m=M, s=S, beta=beta, seed=SEED_BASE + i
            )
            c = clustering(g)
            q = modularity_ground_truth(g)
            comm = community_detection_scores(g)

            c_global_list.append(c["C_global"])
            c_local_list.append(c["C_local"])
            q_list.append(q)
            nmi_l_list.append(comm["nmi_louvain"])
            ari_l_list.append(comm["ari_louvain"])
            nmi_ld_list.append(comm["nmi_leiden"])
            ari_ld_list.append(comm["ari_leiden"])

        rows.append(
            {
                "beta": beta,
                "C_global_mean": np.mean(c_global_list),
                "C_global_std": np.std(c_global_list),
                "C_local_mean": np.mean(c_local_list),
                "C_local_std": np.std(c_local_list),
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
            f"beta={beta:<5} C_global={np.mean(c_global_list):.4f}  "
            f"Q={np.mean(q_list):.4f}  NMI(Leiden)={np.mean(nmi_ld_list):.4f}"
        )

    return rows


def plot_clustering_vs_beta(rows, out_path_png):
    betas = [r["beta"] for r in rows]
    c_global = [r["C_global_mean"] for r in rows]
    c_global_std = [r["C_global_std"] for r in rows]
    c_local = [r["C_local_mean"] for r in rows]
    c_local_std = [r["C_local_std"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        betas,
        c_global,
        yerr=c_global_std,
        marker="o",
        capsize=3,
        label="Tranzytywność (C_globalny)",
    )
    plt.errorbar(
        betas,
        c_local,
        yerr=c_local_std,
        marker="s",
        capsize=3,
        label="Średnia lokalna (C_lokalny)",
    )
    plt.xlabel("β")
    plt.ylabel("Współczynnik gronowania")
    plt.title("Współczynnik gronowania w funkcji β")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_modularity_vs_beta(rows, out_path_png):
    betas = [r["beta"] for r in rows]
    q = [r["Q_mean"] for r in rows]
    q_std = [r["Q_std"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.errorbar(betas, q, yerr=q_std, marker="o", capsize=3, color="#2ca02c")
    plt.xlabel("β")
    plt.ylabel("Modularność względem podziału rzeczywistego Q")
    plt.title("Modularność względem podziału rzeczywistego w funkcji β")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_detectability_vs_beta(rows, out_path_png):
    betas = [r["beta"] for r in rows]
    nmi_l = [r["nmi_louvain_mean"] for r in rows]
    nmi_l_std = [r["nmi_louvain_std"] for r in rows]
    ari_l = [r["ari_louvain_mean"] for r in rows]
    ari_l_std = [r["ari_louvain_std"] for r in rows]
    nmi_ld = [r["nmi_leiden_mean"] for r in rows]
    nmi_ld_std = [r["nmi_leiden_std"] for r in rows]
    ari_ld = [r["ari_leiden_mean"] for r in rows]
    ari_ld_std = [r["ari_leiden_std"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        betas, nmi_l, yerr=nmi_l_std, marker="o", capsize=3, label="NMI (Louvain)"
    )
    plt.errorbar(
        betas,
        ari_l,
        yerr=ari_l_std,
        marker="o",
        capsize=3,
        linestyle="--",
        label="ARI (Louvain)",
    )
    plt.errorbar(
        betas, nmi_ld, yerr=nmi_ld_std, marker="s", capsize=3, label="NMI (Leiden)"
    )
    plt.errorbar(
        betas,
        ari_ld,
        yerr=ari_ld_std,
        marker="s",
        capsize=3,
        linestyle="--",
        label="ARI (Leiden)",
    )
    plt.xlabel("β")
    plt.ylabel("Zgodność z podziałem rzeczywistym")
    plt.title("Wykrywalność struktury społecznościowej w funkcji β")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print(f"Sweep beta: N={N}, K={K}, m={M}, {NUM_RUNS} przebiegow na punkt...\n")
    rows = run_sweep()

    plot_clustering_vs_beta(rows, "outputs/clustering_vs_beta.png")
    plot_modularity_vs_beta(rows, "outputs/modularnosc_vs_beta.png")
    plot_detectability_vs_beta(rows, "outputs/wykrywalnosc_vs_beta.png")

    print(
        "\nZapisano: clustering_vs_beta.png, modularnosc_vs_beta.png, "
        "wykrywalnosc_vs_beta.png"
    )

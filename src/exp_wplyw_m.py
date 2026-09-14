import numpy as np
import matplotlib.pyplot as plt

from src.modele import generate_own_model
from src.metryki import (
    clustering,
    modularity_ground_truth,
    community_detection_scores,
    fit_and_compare_power_law,
)

N = 1000
K = 5
M0 = 3
S = [0.2] * K
BETA = 0.1
M_VALUES = [1, 2, 3, 5, 8, 12]
NUM_RUNS = 100
SEED_BASE = 0


def run_sweep():
    rows = []
    for m in M_VALUES:
        m0 = min(M0, m)
        avg_deg_list, gamma_list, c_list, q_list, nmi_list = [], [], [], [], []

        for i in range(NUM_RUNS):
            g = generate_own_model(
                N=N, K=K, m0=max(m0, m), m=m, s=S, beta=BETA, seed=SEED_BASE + i
            )
            avg_deg_list.append(2 * g.ecount() / g.vcount())
            try:
                _, stats = fit_and_compare_power_law(g)
                gamma_list.append(stats["gamma"])
            except Exception:
                pass
            c_list.append(clustering(g)["C_global"])
            q_list.append(modularity_ground_truth(g))
            nmi_list.append(community_detection_scores(g)["nmi_leiden"])

        rows.append(
            {
                "m": m,
                "avg_degree_mean": np.mean(avg_deg_list),
                "avg_degree_std": np.std(avg_deg_list),
                "gamma_mean": np.mean(gamma_list) if gamma_list else np.nan,
                "gamma_std": np.std(gamma_list) if gamma_list else np.nan,
                "C_global_mean": np.mean(c_list),
                "C_global_std": np.std(c_list),
                "Q_mean": np.mean(q_list),
                "Q_std": np.std(q_list),
                "nmi_mean": np.mean(nmi_list),
                "nmi_std": np.std(nmi_list),
            }
        )
        print(
            f"m={m:<3} avg_degree={np.mean(avg_deg_list):.2f}  "
            f"gamma={np.mean(gamma_list):.3f}  C_global={np.mean(c_list):.4f}  "
            f"Q={np.mean(q_list):.3f}  NMI={np.mean(nmi_list):.3f}"
        )

    return rows


def plot_m_sweep(rows, out_path_png):
    m_vals = [r["m"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].errorbar(
        m_vals,
        [r["gamma_mean"] for r in rows],
        yerr=[r["gamma_std"] for r in rows],
        marker="o",
        capsize=3,
        color="#1f77b4",
    )
    axes[0].axhline(
        3.0, linestyle=":", color="gray", label="γ=3 (przewidywanie teoretyczne)"
    )
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("Wykładnik γ")
    axes[0].set_title("Wykładnik potęgowy γ w funkcji m")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].errorbar(
        m_vals,
        [r["C_global_mean"] for r in rows],
        yerr=[r["C_global_std"] for r in rows],
        marker="s",
        capsize=3,
        color="#2ca02c",
    )
    axes[1].set_xlabel("m")
    axes[1].set_ylabel("Współczynnik gronowania C_global")
    axes[1].set_title("Klastrowanie w funkcji m")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    axes[2].errorbar(
        m_vals,
        [r["Q_mean"] for r in rows],
        yerr=[r["Q_std"] for r in rows],
        marker="^",
        capsize=3,
        label="Modularność Q",
        color="#d62728",
    )
    axes[2].errorbar(
        m_vals,
        [r["nmi_mean"] for r in rows],
        yerr=[r["nmi_std"] for r in rows],
        marker="d",
        capsize=3,
        label="NMI (Leiden)",
        color="#9467bd",
    )
    axes[2].set_xlabel("m")
    axes[2].set_ylabel("Wartość metryki")
    axes[2].set_title("Modularność i wykrywalność w funkcji m")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print(f"Sweep m: N={N}, K={K}, beta={BETA}, {NUM_RUNS} przebiegow na punkt...\n")
    rows = run_sweep()
    plot_m_sweep(rows, "outputs/sweep_m_parameter.png")
    print("\nZapisano: sweep_m_parameter.png")

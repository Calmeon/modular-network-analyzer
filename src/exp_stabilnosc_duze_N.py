import numpy as np

from src.modele import generate_own_model
from src.metryki import (
    clustering,
    modularity_ground_truth,
    community_detection_scores,
    fit_and_compare_power_law,
)

K = 5
M0 = 3
M = 3
S = [0.2] * K
BETA = 0.1
SEED_BASE = 0

N_VALUES = [1000, 5000]
NUM_RUNS = {1000: 20, 5000: 10}


def run_for_N(N, num_runs):
    avg_deg_list, gamma_list, c_list, q_list, nmi_list = [], [], [], [], []
    for i in range(num_runs):
        g = generate_own_model(N=N, K=K, m0=M0, m=M, s=S, beta=BETA, seed=SEED_BASE + i)
        avg_deg_list.append(2 * g.ecount() / g.vcount())
        _, stats = fit_and_compare_power_law(g)
        gamma_list.append(stats["gamma"])
        c_list.append(clustering(g)["C_global"])
        q_list.append(modularity_ground_truth(g))
        nmi_list.append(community_detection_scores(g)["nmi_leiden"])

    return {
        "N": N,
        "num_runs": num_runs,
        "avg_degree_mean": np.mean(avg_deg_list),
        "avg_degree_std": np.std(avg_deg_list),
        "gamma_mean": np.mean(gamma_list),
        "gamma_std": np.std(gamma_list),
        "C_global_mean": np.mean(c_list),
        "C_global_std": np.std(c_list),
        "Q_mean": np.mean(q_list),
        "Q_std": np.std(q_list),
        "nmi_mean": np.mean(nmi_list),
        "nmi_std": np.std(nmi_list),
    }


if __name__ == "__main__":
    results = []
    for N in N_VALUES:
        print(f"N={N}: generuje {NUM_RUNS[N]} przebiegow...")
        results.append(run_for_N(N, NUM_RUNS[N]))

    print(f"\n{'N':<8}{'avg_deg':>12}{'gamma':>14}{'C_global':>14}{'Q':>12}{'NMI':>12}")
    for r in results:
        print(
            f"{r['N']:<8}"
            f"{r['avg_degree_mean']:>7.2f}±{r['avg_degree_std']:<4.2f}"
            f"{r['gamma_mean']:>8.3f}±{r['gamma_std']:<4.3f}"
            f"{r['C_global_mean']:>8.4f}±{r['C_global_std']:<4.4f}"
            f"{r['Q_mean']:>7.3f}±{r['Q_std']:<4.3f}"
            f"{r['nmi_mean']:>7.3f}±{r['nmi_std']:<4.3f}"
        )

    with open("outputs/tabela_robustness_N.md", "w", encoding="utf-8") as f:
        f.write("# Test robustności - stabilność metryk przy większym N\n\n")
        f.write(f"Parametry stałe: K={K}, m={M}, s={S}, β={BETA}\n\n")
        f.write("| N | Liczba przebiegów | Śr. stopień | γ | C_global | Q | NMI |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['N']} | {r['num_runs']} | "
                f"{r['avg_degree_mean']:.2f}±{r['avg_degree_std']:.2f} | "
                f"{r['gamma_mean']:.3f}±{r['gamma_std']:.3f} | "
                f"{r['C_global_mean']:.4f}±{r['C_global_std']:.4f} | "
                f"{r['Q_mean']:.3f}±{r['Q_std']:.3f} | "
                f"{r['nmi_mean']:.3f}±{r['nmi_std']:.3f} |\n"
            )

    print("\nZapisano: tabela_robustness_N.md")

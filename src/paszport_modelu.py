import numpy as np

from src.modele import generate_own_model
from src.metryki import (
    clustering,
    avg_path_length,
    small_worldness_sigma,
    modularity_ground_truth,
    fit_and_compare_power_law,
)

N = 1000
K = 5
M0 = 3
M = 3
S = [0.2] * K
BETA = 0.1
NUM_RUNS = 100
SEED_BASE = 0


def save_markdown_table(rows, path):
    lines = [
        "# Tabela - Zestawienie liczbowe dla baseline\n",
        f"Parametry: N={N}, K={K}, m0={M0}, m={M}, s={S}, β={BETA}, "
        f"{NUM_RUNS} przebiegów\n",
        "| Metryka | Średnia | Odch. std |",
        "|---|---:|---:|",
    ]
    for name, mean, std in rows:
        lines.append(f"| {name} | {mean:.4f} | {std:.4f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    avg_deg_list, gamma_list = [], []
    c_global_list, c_local_list = [], []
    L_list, sigma_list, Q_list = [], [], []

    print(f"Generuje {NUM_RUNS} przebiegow (N={N}, baseline)...")
    for i in range(NUM_RUNS):
        g = generate_own_model(N=N, K=K, m0=M0, m=M, s=S, beta=BETA, seed=SEED_BASE + i)

        avg_deg_list.append(2 * g.ecount() / g.vcount())

        _, stats = fit_and_compare_power_law(g)
        gamma_list.append(stats["gamma"])

        c = clustering(g)
        c_global_list.append(c["C_global"])
        c_local_list.append(c["C_local"])

        L_list.append(avg_path_length(g))
        sigma_list.append(small_worldness_sigma(g, n_random_refs=5, seed=SEED_BASE + i))
        Q_list.append(modularity_ground_truth(g))

        print(f"  przebieg {i+1}/{NUM_RUNS} gotowy")

    rows = [
        ("Średni stopień", np.mean(avg_deg_list), np.std(avg_deg_list)),
        ("Wykładnik γ", np.mean(gamma_list), np.std(gamma_list)),
        ("C_global", np.mean(c_global_list), np.std(c_global_list)),
        ("C_local", np.mean(c_local_list), np.std(c_local_list)),
        ("Średnia długość ścieżki L", np.mean(L_list), np.std(L_list)),
        ("Współczynnik small-world σ", np.nanmean(sigma_list), np.nanstd(sigma_list)),
        ("Modularność Q", np.mean(Q_list), np.std(Q_list)),
    ]

    print(f"\n{'Metryka':<28} {'Średnia':>10} {'Odch.std':>10}")
    for name, mean, std in rows:
        print(f"{name:<28} {mean:>10.4f} {std:>10.4f}")

    save_markdown_table(rows, "outputs/tabela_paszport_modelu.md")
    print("\nZapisano: tabela_paszport_modelu.md")

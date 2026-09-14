from src.modele import generate_own_model
from src.metryki import (
    averaged_power_law_fit,
    plot_averaged_power_law_fit,
    averaged_degree_stats,
    plot_avg_degree_vs_node_index,
)

K = 5
M0 = 3
M = 3
S = [0.2] * K
BETA = 0.1
SEED_BASE = 0

N_TAIL = 5000
N_STRUCT = 1000

NUM_RUNS_TAIL = 100
NUM_RUNS_STRUCT = 100


if __name__ == "__main__":
    print(f"Generuje {NUM_RUNS_TAIL} przebiegow (N={N_TAIL})...")
    params_tail = dict(N=N_TAIL, K=K, m0=M0, m=M, s=S, beta=BETA)

    stats, ccdf_data = averaged_power_law_fit(
        generate_own_model, params_tail, num_runs=NUM_RUNS_TAIL, seed_base=SEED_BASE
    )

    print(f"      gamma = {stats['gamma_mean']:.3f} +/- {stats['gamma_std']:.3f}")
    print(f"      xmin  = {stats['xmin_mean']:.1f} +/- {stats['xmin_std']:.1f}")
    print(f"      KS    = {stats['ks_mean']:.4f} +/- {stats['ks_std']:.4f}")
    print(
        f"      power-law vs exponential: R={stats['R_exp_mean']:+.2f}, "
        f"p={stats['p_exp_mean']:.4f}"
    )
    print(
        f"      power-law vs lognormal  : R={stats['R_ln_mean']:+.2f}, "
        f"p={stats['p_ln_mean']:.4f}"
    )

    plot_averaged_power_law_fit(
        ccdf_data,
        gamma_mean=stats["gamma_mean"],
        xmin_mean=stats["xmin_mean"],
        title=f"Rozkład stopni (CCDF) - N={N_TAIL}, {NUM_RUNS_TAIL} przebiegów",
        out_path_png="outputs/rozklad_stopni_ccdf.png",
    )
    print("      Zapisano: rozklad_stopni_ccdf.png\n")

    print(f"Generuje {NUM_RUNS_STRUCT} przebiegow (N={N_STRUCT})...")
    params_struct = dict(N=N_STRUCT, K=K, m0=M0, m=M, s=S, beta=BETA)

    _, _, avg_degree_per_node = averaged_degree_stats(
        generate_own_model, params_struct, num_runs=NUM_RUNS_STRUCT, seed_base=SEED_BASE
    )

    plot_avg_degree_vs_node_index(
        avg_degree_per_node,
        out_path_png="outputs/sredni_stopien_vs_numer_wezla.png",
        title=f"Średni stopień w funkcji numeru węzła - N={N_STRUCT}, "
        f"{NUM_RUNS_STRUCT} przebiegów",
    )
    print("      Zapisano: sredni_stopien_vs_numer_wezla.png")

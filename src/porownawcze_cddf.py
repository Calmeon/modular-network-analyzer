import numpy as np
import matplotlib.pyplot as plt

from src.modele import (
    generate_own_model,
    generate_ba,
    generate_holme_kim,
    generate_erdos_renyi,
    generate_sbm,
    generate_lfr,
)
from src.metryki import (
    clustering,
    avg_path_length,
    small_worldness_sigma,
    modularity_ground_truth,
    community_detection_scores,
    fit_and_compare_power_law,
)

N = 1000
AVG_DEGREE = 6.53
K = 5
GROUP_SIZE = N // K
SEED_BASE = 0
ND = "N/D"

MODEL_FACTORIES = {
    "Model własny": lambda seed: generate_own_model(
        N=N, K=K, m0=3, m=3, s=[1 / K] * K, beta=0.1, seed=seed
    ),
    "BA": lambda seed: generate_ba(n=N, m=3, seed=seed),
    "Holme-Kim (p=1.0)": lambda seed: generate_holme_kim(n=N, m=3, p=1.0, seed=seed),
    "Erdős-Rényi": lambda seed: generate_erdos_renyi(
        n=N, avg_degree=AVG_DEGREE, seed=seed
    ),
    "SBM": lambda seed: generate_sbm(
        sizes=[GROUP_SIZE] * K, p_in=0.0298, p_out=0.00074, seed=seed
    ),
    "LFR": lambda seed: generate_lfr(
        n=N,
        avg_degree=AVG_DEGREE,
        max_degree=50,
        mu=0.091,
        min_community=150,
        max_community=250,
        seed=seed,
    ),
}

NUM_RUNS = {
    "Model własny": 20,
    "BA": 20,
    "Holme-Kim (p=1.0)": 20,
    "Erdős-Rényi": 20,
    "SBM": 20,
    "LFR": 10,  # LFR wolniejszy i moze nie zbiec
}


def safe_gamma(g):
    try:
        _, stats = fit_and_compare_power_law(g)
        return stats["gamma"]
    except Exception:
        return np.nan


def run_model_repeated(name, factory, num_runs):
    metrics_rows = []
    degree_sequences = []

    for i in range(num_runs):
        g = factory(SEED_BASE + i)
        if g is None:  # LFR czasem nie zbiega
            continue

        has_groups = "group" in g.vertex_attributes()
        avg_deg = 2 * g.ecount() / g.vcount()
        gamma = safe_gamma(g) if name != "Erdős-Rényi" else np.nan
        c = clustering(g)["C_global"]
        L = avg_path_length(g)
        sigma = small_worldness_sigma(g, n_random_refs=3, seed=SEED_BASE + i)
        Q = modularity_ground_truth(g) if has_groups else np.nan
        nmi = community_detection_scores(g)["nmi_leiden"] if has_groups else np.nan

        metrics_rows.append(
            {
                "avg_degree": avg_deg,
                "gamma": gamma,
                "C_global": c,
                "L": L,
                "sigma": sigma,
                "Q": Q,
                "NMI": nmi,
            }
        )
        degree_sequences.append(np.array(g.degree()))

    if not metrics_rows:
        print(f"  [{name}] UWAGA: zaden przebieg sie nie powiodl!")
        return None, []

    n_ok = len(metrics_rows)
    n_failed = num_runs - n_ok
    if n_failed > 0:
        print(
            f"  [{name}] {n_failed}/{num_runs} przebiegow nie powiodlo sie (pominieto)"
        )

    agg = {}
    for key in ["avg_degree", "gamma", "C_global", "L", "sigma", "Q", "NMI"]:
        values = [r[key] for r in metrics_rows if not np.isnan(r[key])]
        agg[key + "_mean"] = np.mean(values) if values else np.nan
        agg[key + "_std"] = np.std(values) if values else np.nan
    agg["n_runs_ok"] = n_ok

    return agg, degree_sequences


def averaged_ccdf(degree_sequences):
    degree_sequences = [d[d > 0] for d in degree_sequences]
    max_degree = max(d.max() for d in degree_sequences)
    degree_values = np.arange(1, max_degree + 1)

    ccdf_matrix = np.zeros((len(degree_sequences), len(degree_values)))
    for i, degrees in enumerate(degree_sequences):
        n = len(degrees)
        sorted_deg = np.sort(degrees)
        idx = np.searchsorted(sorted_deg, degree_values, side="left")
        ccdf_matrix[i] = (n - idx) / n

    return degree_values, ccdf_matrix.mean(axis=0)


def save_markdown_table(all_agg, path):
    def fmt(mean, std):
        if mean is None or np.isnan(mean):
            return ND
        return f"{mean:.4f} ± {std:.4f}"

    lines = [
        "# Tabela - Zbiorcze porównanie metryk strukturalnych (uśrednione)\n",
        "Wartości: średnia ± odchylenie standardowe po powtórzeniach "
        "(liczba przebiegów podana w nawiasie).\n",
        "| Model | n | Śr. stopień | γ | C_global | L | σ | Q | NMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, agg in all_agg.items():
        lines.append(
            f"| {name} | {agg['n_runs_ok']} | "
            f"{fmt(agg['avg_degree_mean'], agg['avg_degree_std'])} | "
            f"{fmt(agg['gamma_mean'], agg['gamma_std'])} | "
            f"{fmt(agg['C_global_mean'], agg['C_global_std'])} | "
            f"{fmt(agg['L_mean'], agg['L_std'])} | "
            f"{fmt(agg['sigma_mean'], agg['sigma_std'])} | "
            f"{fmt(agg['Q_mean'], agg['Q_std'])} | "
            f"{fmt(agg['NMI_mean'], agg['NMI_std'])} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_comparative_ccdf(ccdf_data, out_path_png, highlight="Model własny"):
    plt.figure(figsize=(7.5, 6))
    for name, (degree_values, mean_ccdf) in ccdf_data.items():
        mask = mean_ccdf > 0
        if name == highlight:
            plt.loglog(
                degree_values[mask],
                mean_ccdf[mask],
                color="#d62728",
                linewidth=2.4,
                zorder=10,
                label=name,
            )
        else:
            plt.loglog(
                degree_values[mask],
                mean_ccdf[mask],
                linewidth=1.2,
                alpha=0.8,
                zorder=3,
                label=name,
            )

    plt.xlabel("Stopień węzła k (skala log)")
    plt.ylabel("P(K ≥ k) (skala log)")
    plt.title("Porównawcza CCDF rozkładu stopni (uśredniona po przebiegach)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    all_agg = {}
    ccdf_data = {}

    for name, factory in MODEL_FACTORIES.items():
        num_runs = NUM_RUNS[name]
        print(f"Generuje {name} ({num_runs} przebiegow)...")
        agg, degree_sequences = run_model_repeated(name, factory, num_runs)
        if agg is None:
            continue
        all_agg[name] = agg
        ccdf_data[name] = averaged_ccdf(degree_sequences)

        print(
            f"  avg_degree={agg['avg_degree_mean']:.3f}±{agg['avg_degree_std']:.3f}  "
            f"C_global={agg['C_global_mean']:.4f}±{agg['C_global_std']:.4f}  "
            f"Q={agg['Q_mean']:.3f}±{agg['Q_std']:.3f}  "
            f"NMI={agg['NMI_mean']:.3f}±{agg['NMI_std']:.3f}"
        )

    save_markdown_table(all_agg, "outputs/tabela_porownanie_modeli.md")
    plot_comparative_ccdf(ccdf_data, "outputs/porownawcza_ccdf.png")
    print("\nZapisano: tabela_porownanie_modeli.md, porownawcza_ccdf.png")

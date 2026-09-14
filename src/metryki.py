import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import powerlaw
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def gini_coefficient(values):
    x = np.sort(np.array(values, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])


def avg_path_length(g):
    giant = g.connected_components().giant()
    return giant.average_path_length()


def small_worldness_sigma(g, n_random_refs=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = g.vcount()
    e = g.ecount()
    C = g.transitivity_avglocal_undirected(mode="zero")
    L = avg_path_length(g)

    c_rand_list, l_rand_list = [], []
    for i in range(n_random_refs):
        g_rand = ig.Graph.Erdos_Renyi(n=n, m=e)
        c_rand_list.append(g_rand.transitivity_avglocal_undirected(mode="zero"))
        try:
            l_rand_list.append(avg_path_length(g_rand))
        except Exception:
            continue

    C_rand = np.mean(c_rand_list) if c_rand_list else np.nan
    L_rand = np.mean(l_rand_list) if l_rand_list else np.nan

    if C_rand == 0 or L_rand == 0 or np.isnan(C_rand) or np.isnan(L_rand):
        return np.nan
    return (C / C_rand) / (L / L_rand)


def clustering(g):
    return {
        "C_global": g.transitivity_undirected(mode="zero"),
        "C_local": g.transitivity_avglocal_undirected(mode="zero"),
    }


def modularity_ground_truth(g):
    return g.modularity(g.vs["group"])


def community_detection_scores(g):
    ground_truth = g.vs["group"]

    louvain = g.community_multilevel()
    leiden = g.community_leiden(objective_function="modularity", n_iterations=10)

    return {
        "modularity_louvain": louvain.modularity,
        "modularity_leiden": g.modularity(leiden.membership),
        "nmi_louvain": normalized_mutual_info_score(ground_truth, louvain.membership),
        "ari_louvain": adjusted_rand_score(ground_truth, louvain.membership),
        "nmi_leiden": normalized_mutual_info_score(ground_truth, leiden.membership),
        "ari_leiden": adjusted_rand_score(ground_truth, leiden.membership),
    }


def degree_distribution_fit(g):
    degrees = np.array(g.degree())
    degrees = degrees[degrees > 0]

    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    return {
        "gamma": fit.power_law.alpha,
        "xmin": fit.power_law.xmin,
        "ks_statistic": fit.power_law.D,
        "avg_degree": float(np.mean(g.degree())),
    }


def fit_and_compare_power_law(g):
    degrees = np.array(g.degree())
    degrees = degrees[degrees > 0]

    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")

    stats = {
        "gamma": fit.power_law.alpha,
        "xmin": fit.power_law.xmin,
        "ks_statistic": fit.power_law.D,
        "R_vs_exponential": R_exp,
        "p_vs_exponential": p_exp,
        "R_vs_lognormal": R_ln,
        "p_vs_lognormal": p_ln,
    }
    return fit, stats


def averaged_power_law_fit(generator_fn, params, num_runs=100, seed_base=0):
    gammas, xmins, kss = [], [], []
    R_exps, p_exps, R_lns, p_lns = [], [], [], []
    degree_sequences = []
    max_degree_overall = 0

    for i in range(num_runs):
        g = generator_fn(**params, seed=seed_base + i)
        degrees = np.array(g.degree())
        degrees = degrees[degrees > 0]
        degree_sequences.append(degrees)
        max_degree_overall = max(max_degree_overall, degrees.max())

        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        gammas.append(fit.power_law.alpha)
        xmins.append(fit.power_law.xmin)
        kss.append(fit.power_law.D)
        R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
        R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
        R_exps.append(R_exp)
        p_exps.append(p_exp)
        R_lns.append(R_ln)
        p_lns.append(p_ln)

    stats_summary = {
        "gamma_mean": np.mean(gammas),
        "gamma_std": np.std(gammas),
        "xmin_mean": np.mean(xmins),
        "xmin_std": np.std(xmins),
        "ks_mean": np.mean(kss),
        "ks_std": np.std(kss),
        "R_exp_mean": np.mean(R_exps),
        "R_exp_std": np.std(R_exps),
        "p_exp_mean": np.mean(p_exps),
        "R_ln_mean": np.mean(R_lns),
        "R_ln_std": np.std(R_lns),
        "p_ln_mean": np.mean(p_lns),
    }

    degree_values = np.arange(1, max_degree_overall + 1)
    ccdf_matrix = np.zeros((num_runs, len(degree_values)))
    for i, degrees in enumerate(degree_sequences):
        n = len(degrees)
        sorted_deg = np.sort(degrees)
        idx = np.searchsorted(sorted_deg, degree_values, side="left")
        ccdf_matrix[i] = (n - idx) / n

    mean_ccdf = ccdf_matrix.mean(axis=0)
    std_ccdf = ccdf_matrix.std(axis=0)

    return stats_summary, (degree_values, mean_ccdf, std_ccdf)


def plot_averaged_power_law_fit(ccdf_data, gamma_mean, xmin_mean, title, out_path_png):
    degree_values, mean_ccdf, std_ccdf = ccdf_data
    mask = mean_ccdf > 0
    x = degree_values[mask]
    y = mean_ccdf[mask]
    y_lo = np.clip(y - std_ccdf[mask], 1e-6, None)
    y_hi = y + std_ccdf[mask]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(
        x,
        y,
        color="black",
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Średnia CCDF",
    )
    ax.fill_between(
        x, y_lo, y_hi, color="black", alpha=0.15, label="± odchylenie standardowe"
    )

    xmin_idx = np.searchsorted(degree_values, xmin_mean)
    y_anchor = mean_ccdf[min(xmin_idx, len(mean_ccdf) - 1)]
    x_line = degree_values[degree_values >= xmin_mean]
    y_line = y_anchor * (x_line / xmin_mean) ** (-(gamma_mean - 1))
    ax.loglog(
        x_line,
        y_line,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Dopasowanie potęgowe (γ={gamma_mean:.2f})",
    )

    ax.set_xlabel("Stopień węzła k (skala log)")
    ax.set_ylabel("P(K ≥ k) (skala log)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


def averaged_degree_distribution(generator_fn, params, num_runs=30, seed_base=0):
    from collections import Counter

    max_degree_overall = 0
    counts_list = []
    n_nodes_list = []

    for i in range(num_runs):
        g = generator_fn(**params, seed=seed_base + i)
        degrees = g.degree()
        n_nodes_list.append(len(degrees))
        counts = Counter(degrees)
        counts_list.append(counts)
        max_degree_overall = max(max_degree_overall, max(degrees))

    degree_values = np.arange(0, max_degree_overall + 1)
    prob_matrix = np.zeros((num_runs, len(degree_values)))
    for i, counts in enumerate(counts_list):
        n = n_nodes_list[i]
        for d in degree_values:
            prob_matrix[i, d] = counts.get(d, 0) / n

    mean_prob = prob_matrix.mean(axis=0)
    std_prob = prob_matrix.std(axis=0)
    return degree_values, mean_prob, std_prob


def averaged_degree_stats(generator_fn, params, num_runs=30, seed_base=0):
    from collections import Counter

    all_degrees_per_node = []
    counts_list = []
    max_degree_overall = 0

    for i in range(num_runs):
        g = generator_fn(**params, seed=seed_base + i)
        degrees = g.degree()
        all_degrees_per_node.append(degrees)
        counts_list.append(Counter(degrees))
        max_degree_overall = max(max_degree_overall, max(degrees))

    degree_values = np.arange(0, max_degree_overall + 1)
    avg_counts = np.array(
        [np.mean([counts.get(d, 0) for counts in counts_list]) for d in degree_values]
    )
    avg_degree_per_node = np.mean(np.array(all_degrees_per_node), axis=0)

    return degree_values, avg_counts, avg_degree_per_node


def plot_avg_count_vs_degree(
    degree_values,
    avg_counts,
    out_path_png,
    title="Średnia liczba węzłów w funkcji stopnia",
):
    """Wykres: os X = stopień węzła k, os Y = średnia liczba węzłów o tym stopniu."""
    plt.figure(figsize=(7, 5))
    plt.plot(
        degree_values,
        avg_counts,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="#d62728",
    )
    plt.xlabel("Stopień węzła k")
    plt.ylabel("Średnia liczba węzłów")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_avg_degree_vs_node_index(
    avg_degree_per_node,
    out_path_png,
    title="Średnia liczba sąsiadów w funkcji numeru węzła",
):
    """Wykres: os X = numer węzła (kolejność dodania), os Y = średni stopień."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(len(avg_degree_per_node)),
        avg_degree_per_node,
        linewidth=1.0,
        color="#d62728",
    )
    plt.xlabel("Numer węzła (kolejność dodania do sieci)")
    plt.ylabel("Średnia liczba sąsiadów (stopień)")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_averaged_degree_distributions(series: dict, out_path_png, highlight=None):
    plt.figure(figsize=(7, 5.5))

    for name, (degree_values, mean_prob, std_prob) in series.items():
        mask = mean_prob > 0
        x = degree_values[mask]
        y = mean_prob[mask]
        y_lo = np.clip(y - std_prob[mask], 1e-6, None)
        y_hi = y + std_prob[mask]

        is_highlight = name == highlight
        color = "#d62728" if is_highlight else None
        lw = 2.4 if is_highlight else 1.2
        alpha_line = 1.0 if is_highlight else 0.8
        zorder = 10 if is_highlight else 3

        (line,) = plt.loglog(
            x,
            y,
            marker="o",
            markersize=4 if is_highlight else 3,
            linewidth=lw,
            alpha=alpha_line,
            zorder=zorder,
            color=color,
            label=name,
        )
        plt.fill_between(
            x, y_lo, y_hi, color=line.get_color(), alpha=0.15, zorder=zorder - 1
        )

    plt.xlabel("Stopień węzła k (skala log)")
    plt.ylabel("P(k) (skala log)")
    plt.title("Uśredniony rozkład stopni węzłów (± odchylenie standardowe)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()

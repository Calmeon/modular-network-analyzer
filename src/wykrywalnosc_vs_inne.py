import numpy as np
import matplotlib.pyplot as plt

from src.modele import (
    generate_own_model,
    generate_ba,
    generate_erdos_renyi,
    generate_sbm,
    generate_lfr,
)
from src.metryki import (
    clustering,
    avg_path_length,
    modularity_ground_truth,
    community_detection_scores,
    fit_and_compare_power_law,
)

N = 1000
K = 5
GROUP_SIZE = N // K
AVG_DEGREE = 6.53
SEED_BASE = 0
NUM_RUNS = 100


def sweep_own_model(beta_values):
    points = []
    for beta in beta_values:
        q_list, nmi_list = [], []
        for i in range(NUM_RUNS):
            g = generate_own_model(
                N=N, K=K, m0=3, m=3, s=[1 / K] * K, beta=beta, seed=SEED_BASE + i
            )
            q_list.append(modularity_ground_truth(g))
            nmi_list.append(community_detection_scores(g)["nmi_leiden"])
        points.append((np.mean(q_list), np.mean(nmi_list)))
    return points


def sweep_lfr(mu_values):
    points = []
    for mu in mu_values:
        q_list, nmi_list = [], []
        for i in range(NUM_RUNS // 2):
            g = generate_lfr(
                n=N,
                avg_degree=AVG_DEGREE,
                max_degree=50,
                mu=mu,
                min_community=150,
                max_community=250,
                seed=SEED_BASE + i,
            )
            if g is None:
                continue
            q_list.append(modularity_ground_truth(g))
            nmi_list.append(community_detection_scores(g)["nmi_leiden"])
        if q_list:
            points.append((np.mean(q_list), np.mean(nmi_list)))
        else:
            print(f"  [LFR] mu={mu}: wszystkie przebiegi nie zbiegly - pomijam punkt")
    return points


def sweep_sbm(p_out_values, p_in=0.0298):
    points = []
    for p_out in p_out_values:
        q_list, nmi_list = [], []
        for i in range(NUM_RUNS):
            g = generate_sbm(
                sizes=[GROUP_SIZE] * K, p_in=p_in, p_out=p_out, seed=SEED_BASE + i
            )
            q_list.append(modularity_ground_truth(g))
            nmi_list.append(community_detection_scores(g)["nmi_leiden"])
        points.append((np.mean(q_list), np.mean(nmi_list)))
    return points


def plot_q_vs_nmi(own_points, lfr_points, sbm_points, out_path_png):
    plt.figure(figsize=(7, 6))

    own_q, own_nmi = zip(*own_points)
    plt.plot(
        own_q,
        own_nmi,
        marker="o",
        linewidth=2,
        color="#d62728",
        label="Model własny (w zależności od β)",
        zorder=5,
    )

    if lfr_points:
        lfr_q, lfr_nmi = zip(*lfr_points)
        plt.plot(
            lfr_q,
            lfr_nmi,
            marker="s",
            linewidth=1.5,
            color="#1f77b4",
            label="LFR (w zależności od μ)",
            zorder=4,
        )

    sbm_q, sbm_nmi = zip(*sbm_points)
    plt.scatter(
        sbm_q,
        sbm_nmi,
        marker="^",
        s=70,
        color="#2ca02c",
        label="SBM (punkty niepołączone)",
        zorder=6,
    )

    plt.xlabel("Modularność względem podziału rzeczywistego Q")
    plt.ylabel("NMI (Leiden) względem podziału rzeczywistego")
    plt.title("Modularność vs wykrywalność struktury społecznościowej")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


def compute_radar_metrics():
    graphs = {
        "Model własny": generate_own_model(
            N=N, K=K, m0=3, m=3, s=[1 / K] * K, beta=0.1, seed=SEED_BASE
        ),
        "BA": generate_ba(n=N, m=3, seed=SEED_BASE),
        "Erdős-Rényi": generate_erdos_renyi(n=N, avg_degree=AVG_DEGREE, seed=SEED_BASE),
        "SBM": generate_sbm(
            sizes=[GROUP_SIZE] * K, p_in=0.0298, p_out=0.00074, seed=SEED_BASE
        ),
    }
    lfr = generate_lfr(
        n=N,
        avg_degree=AVG_DEGREE,
        max_degree=50,
        mu=0.091,
        min_community=150,
        max_community=250,
        seed=SEED_BASE,
    )
    if lfr is not None:
        graphs["LFR"] = lfr

    raw = {}
    for name, g in graphs.items():
        has_groups = "group" in g.vertex_attributes()
        c = clustering(g)["C_global"]
        L = avg_path_length(g)

        try:
            _, stats = fit_and_compare_power_law(g)
            gamma_score = max(0.0, 1 - abs(stats["gamma"] - 3) / 3)
        except Exception:
            gamma_score = 0.0

        Q = modularity_ground_truth(g) if has_groups else 0.0
        nmi = community_detection_scores(g)["nmi_leiden"] if has_groups else 0.0

        raw[name] = {
            "C": c,
            "invL": 1 / L,
            "Q": Q,
            "gamma_score": gamma_score,
            "NMI": nmi,
        }

    axes = ["C", "invL", "Q", "gamma_score", "NMI"]
    normalized = {name: {} for name in raw}
    for axis in axes:
        values = [raw[name][axis] for name in raw]
        lo, hi = min(values), max(values)
        for name in raw:
            normalized[name][axis] = (
                (raw[name][axis] - lo) / (hi - lo) if hi > lo else 0.5
            )

    return normalized, axes


def plot_radar(normalized, axes, out_path_png):
    labels = [
        "Klastrowanie (C)",
        "Krótkie ścieżki (1/L)",
        "Modularność (Q)",
        "Zgodność γ z teorią",
        "Wykrywalność (NMI)",
    ]
    num_axes = len(axes)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    other_colors = {
        "BA": "#ff7f0e",
        "Erdős-Rényi": "#2ca02c",
        "SBM": "#9467bd",
        "LFR": "#8c564b",
    }

    for name, values in normalized.items():
        data = [values[axis] for axis in axes]
        data += data[:1]
        is_own = name == "Model własny"
        color = "#d62728" if is_own else other_colors.get(name, "#7f7f7f")
        ax.plot(
            angles,
            data,
            linewidth=2.4 if is_own else 1.3,
            color=color,
            label=name,
            zorder=10 if is_own else 3,
        )
        ax.fill(angles, data, alpha=0.15 if is_own else 0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Znormalizowany profil modeli (baseline)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=== Wykres: Q vs NMI ===")
    beta_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    mu_values = [0.045, 0.091, 0.167, 0.231, 0.333, 0.429, 0.5]
    p_out_values = [0.0003, 0.0007, 0.0015, 0.003]

    print("Sweep: model wlasny...")
    own_points = sweep_own_model(beta_values)
    print("Sweep: LFR...")
    lfr_points = sweep_lfr(mu_values)
    print("Sweep: SBM...")
    sbm_points = sweep_sbm(p_out_values)

    plot_q_vs_nmi(
        own_points,
        lfr_points,
        sbm_points,
        "outputs/modularnosc_vs_wykrywalnosc.png",
    )
    print("Zapisano: modularnosc_vs_wykrywalnosc.png\n")

    print("=== Wykres: radar ===")
    normalized, axes = compute_radar_metrics()
    plot_radar(normalized, axes, "outputs/radar_profil_modeli.png")
    print("Zapisano: radar_profil_modeli.png")

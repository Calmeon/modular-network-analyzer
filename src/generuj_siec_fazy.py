import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from src.modele import (
    generate_own_model_phase1,
    generate_own_model_phase2,
    round_percentages_to_100,
)

RNG_SEED = 7
CMAP = plt.get_cmap("Set1")
MIN_SIZE, MAX_SIZE = 20, 100


def plot_phase(g, coords, title, out_path_png):
    groups = np.array(g.vs["group"])
    unique_groups = sorted(set(groups))
    group_color = {
        grp: mcolors.to_hex(CMAP(i % 9)) for i, grp in enumerate(unique_groups)
    }
    node_colors = [group_color[grp] for grp in groups]

    n_total = len(groups)
    raw_pct = {grp: 100 * np.sum(groups == grp) / n_total for grp in unique_groups}
    group_share_pct = round_percentages_to_100(raw_pct)

    degrees = np.array(g.degree())
    max_deg = degrees.max() if degrees.max() > 0 else 1
    sizes = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * (degrees / max_deg)

    fig, ax = plt.subplots(figsize=(8, 8))

    segments = [(coords[u], coords[v]) for u, v in g.get_edgelist()]
    lc = LineCollection(segments, colors="#BBBBBB", linewidths=0.5, alpha=0.6, zorder=1)
    ax.add_collection(lc)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=sizes,
        c=node_colors,
        edgecolors="black",
        linewidths=0.6,
        zorder=2,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=group_color[g_id],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=9,
            label=f"Grupa {g_id} ({group_share_pct[g_id]:.0f}%)",
        )
        for g_id in unique_groups
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
        title="Przynależność do grupy\n(udział w sieci)",
        title_fontsize=9,
    )

    # ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    N, K, m0, m = 150, 4, 3, 2
    s = [0.35, 0.30, 0.20, 0.15]
    beta = 0.06

    G_phase1 = generate_own_model_phase1(N=N, K=K, m0=m0, m=m, s=s, seed=RNG_SEED)
    print(
        f"Po Fazie I: N={G_phase1.vcount()}, E={G_phase1.ecount()}, "
        f"liczba spojnych skladowych={len(G_phase1.connected_components())}"
    )

    G_phase2 = generate_own_model_phase2(G_phase1.copy(), beta=beta, N=N, seed=RNG_SEED)
    coords = np.array(G_phase2.layout("fr").coords)

    print(
        f"Po Fazie II: N={G_phase2.vcount()}, E={G_phase2.ecount()}, "
        f"liczba spojnych skladowych={len(G_phase2.connected_components())}"
    )

    output_folder = "outputs"
    plot_phase(
        G_phase1,
        coords,
        title=f"Sieć po Fazie I: wzrost wewnątrzgrupowy (N={N}, K={K}, m={m})",
        out_path_png=f"{output_folder}/siec_faza1.png",
    )
    plot_phase(
        G_phase2,
        coords,
        title=f"Sieć po Fazie II: po dodaniu połączeń międzygrupowych (β={beta})",
        out_path_png=f"{output_folder}/siec_faza2.png",
    )
    print("Zapisano: siec_faza1.png, siec_faza2.png")

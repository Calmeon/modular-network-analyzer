import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import igraph as ig

from src.modele import generate_own_model, compute_group_sizes, round_percentages_to_100

RNG_SEED = 7
CMAP = mpl.colormaps["Set1"]


def plot_network(g, title, out_path_png, seed=None):
    if seed is not None:
        np.random.seed(seed)

    degrees = g.degree()
    max_degree = max(degrees) if max(degrees) > 0 else 1
    min_size, max_size = 8, 65
    vertex_sizes = [
        min_size + (d / max_degree) * (max_size - min_size) for d in degrees
    ]

    groups = g.vs["group"]
    unique_groups = sorted(set(groups))
    vertex_colors = [
        mcolors.to_hex(CMAP(unique_groups.index(grp) % 9)) for grp in groups
    ]

    n_total = len(groups)
    raw_pct = {grp: 100 * groups.count(grp) / n_total for grp in unique_groups}
    group_share_pct = round_percentages_to_100(raw_pct)

    layout = g.layout_fruchterman_reingold()

    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        g,
        target=ax,
        layout=layout,
        vertex_size=vertex_sizes,
        vertex_color=vertex_colors,
        vertex_frame_width=0.7,
        vertex_frame_color="black",
        edge_width=0.4,
        edge_color="#BBBBBB",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=mcolors.to_hex(CMAP(i % 9)),
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=9,
            label=f"Grupa {g_id} ({group_share_pct[g_id]:.0f}%)",
        )
        for i, g_id in enumerate(unique_groups)
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
        title="Przynależność do grupy\n(udział w sieci)",
        title_fontsize=9,
    )

    ax.set_title(title, fontsize=13)
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

    n_i = compute_group_sizes(N, s)
    print("Rozklad wielkosci grup:")
    for group_id, (share, n) in enumerate(zip(s, n_i)):
        print(f"  Grupa {group_id}: udzial s={share:.2f}  ->  n={n} wezlow")
    print()

    G = generate_own_model(N=N, K=K, m0=m0, m=m, s=s, beta=beta, seed=RNG_SEED)

    print(
        f"Wygenerowano graf: N={G.vcount()}, E={G.ecount()}, "
        f"sredni stopien={2*G.ecount()/G.vcount():.2f}"
    )
    print(
        f"Modularnosc względem podziału rzeczywistego: {G.modularity(G.vs['group']):.3f}"
    )

    title = f"Wizualizacja wygenerowanej sieci (N={N}, K={K}, m={m}, β={beta})"
    output_folder = "outputs"
    plot_network(
        G,
        title,
        out_path_png=f"{output_folder}/wlasny_model_wizualizacja.png",
        seed=RNG_SEED,
    )
    print("Zapisano: wlasny_model_wizualizacja.png")

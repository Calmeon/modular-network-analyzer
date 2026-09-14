import numpy as np
import igraph as ig
import networkx as nx


def skewed_group_shares(K, skew):
    ranks = np.arange(1, K + 1)
    weights = ranks.astype(float) ** (-skew)
    return (weights / weights.sum()).tolist()


def compute_group_sizes(N, s):
    s_norm = [x / sum(s) for x in s]
    n_i = [int(N * share) for share in s_norm]
    n_i[0] += N - sum(n_i)
    return n_i


def round_percentages_to_100(pct_dict):
    keys = list(pct_dict.keys())
    raw = np.array([pct_dict[k] for k in keys], dtype=float)

    floor_vals = np.floor(raw).astype(int)
    remainders = raw - floor_vals
    deficit = 100 - floor_vals.sum()

    order = np.argsort(-remainders)
    result = floor_vals.copy()
    for i in range(int(deficit)):
        result[order[i]] += 1

    return {k: int(v) for k, v in zip(keys, result)}


def generate_own_model_phase1(N, K, m0, m, s, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if len(s) != K:
        raise ValueError("Liczba udzialow w 's' musi byc rowna K.")

    n_i = compute_group_sizes(N, s)

    subgraphs = []
    for group_id, n in enumerate(n_i):
        if n < m0:
            raise ValueError(
                f"Grupa {group_id} ma za malo wezlow ({n}) wzgledem m0={m0}."
            )
        clique = ig.Graph.Full(m0)
        g_i = ig.Graph.Barabasi(n=n, m=m, start_from=clique, directed=False)
        g_i.vs["group"] = group_id
        subgraphs.append(g_i)

    return subgraphs[0].disjoint_union(subgraphs[1:])


def generate_own_model_phase2(G, beta, N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    intra_edges = G.ecount()
    M_inter = int(beta * intra_edges)

    degrees = np.array(G.degree())
    all_nodes = np.arange(N)
    existing_edges = set(G.get_edgelist())
    edges_to_add = []
    added = 0

    while added < M_inter:
        total_degree = degrees.sum()
        if total_degree == 0:
            break
        p = degrees / total_degree
        v1, v2 = np.random.choice(all_nodes, size=2, replace=False, p=p)
        if v1 > v2:
            v1, v2 = v2, v1
        if G.vs[v1]["group"] != G.vs[v2]["group"] and (v1, v2) not in existing_edges:
            edges_to_add.append((v1, v2))
            existing_edges.add((v1, v2))
            degrees[v1] += 1
            degrees[v2] += 1
            added += 1

    G.add_edges(edges_to_add)
    return G


def generate_own_model(N, K, m0, m, s, beta, seed=None):
    G = generate_own_model_phase1(N, K, m0, m, s, seed=seed)
    G = generate_own_model_phase2(G, beta, N, seed=seed)
    return G


def generate_erdos_renyi(n, avg_degree, seed=None):
    p = avg_degree / (n - 1)
    G_nx = nx.gnp_random_graph(n=n, p=p, seed=seed)
    return ig.Graph.from_networkx(G_nx)


def generate_sbm(sizes, p_in, p_out, seed=None):
    k = len(sizes)
    prob_matrix = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G_nx = nx.stochastic_block_model(sizes, prob_matrix, seed=seed)
    g = ig.Graph.from_networkx(G_nx)
    group = []
    for gid, size in enumerate(sizes):
        group += [gid] * size
    g.vs["group"] = group
    return g


def generate_lfr(
    n,
    avg_degree,
    max_degree,
    mu,
    min_community,
    max_community,
    tau1=3,
    tau2=1.5,
    seed=None,
):
    try:
        G_nx = nx.LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=avg_degree,
            max_degree=max_degree,
            min_community=min_community,
            max_community=max_community,
            seed=seed,
            max_iters=500,
        )
    except Exception as e:
        print(f"  [LFR] Nie udalo sie zbiec (seed={seed}): {e}")
        return None

    g = ig.Graph.from_networkx(G_nx)
    communities = {frozenset(G_nx.nodes[v]["community"]) for v in G_nx}
    node_group = {}
    for gid, comm in enumerate(communities):
        for node in comm:
            node_group[node] = gid
    g.vs["group"] = [node_group[v] for v in range(g.vcount())]
    return g


def generate_ba(n, m, seed=None):
    G_nx = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return ig.Graph.from_networkx(G_nx)


def generate_holme_kim(n, m, p, seed=None):
    G_nx = nx.powerlaw_cluster_graph(n=n, m=m, p=p, seed=seed)
    return ig.Graph.from_networkx(G_nx)

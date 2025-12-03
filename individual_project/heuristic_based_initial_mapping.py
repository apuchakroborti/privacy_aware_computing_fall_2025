import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------------------
# Load graphs
# ------------------------------------------------------------
def load_graph(edge_file):
    G = nx.read_edgelist(edge_file, nodetype=int)
    return G


# ------------------------------------------------------------
# Compute structural features (degree, clustering, pagerank)
# ------------------------------------------------------------
def compute_features(G):
    degree = dict(G.degree())
    # for item in degree.items():
    #     print(f'Node: {item.key}, Degree: {item.value}')
    print(f'Degree: {degree}')
    cluster = nx.clustering(G)
    # for item in degree.items():
    print(f'Cluster: {cluster}')

    pagerank = nx.pagerank(G, alpha=0.85)
    print(f'PageRank: {pagerank}')

    # Combine into 3-dim feature vector
    features = {}
    for node in G.nodes():
        features[node] = np.array([
            degree[node],
            cluster[node],
            pagerank[node]
        ])
    return features


# ------------------------------------------------------------
# Normalize features to avoid scale dominance
# ------------------------------------------------------------
def normalize_features(features_dict):
    nodes = list(features_dict.keys())
    X = np.array([features_dict[n] for n in nodes])
    print(f'From feature dictionary to numpy array: {X}')
    # Standard normalization
    mean = X.mean(axis=0)
    print(f'Mean of X: {mean}')
    
    std = X.std(axis=0) + 1e-9  # avoid division by zero
    print(f'Standard deviation of X, std: {std}')

    X_norm = (X - mean) / std
    print(f'Normalized features of X, X_norm: {X_norm}')

    # Return dictionary
    return {node: vec for node, vec in zip(nodes, X_norm)}


# ------------------------------------------------------------
# Compute similarity matrix between G1 and G2
# ------------------------------------------------------------
def build_similarity(features1, features2):
    nodes1 = list(features1.keys())
    nodes2 = list(features2.keys())

    X = np.array([features1[n] for n in nodes1])
    Y = np.array([features2[n] for n in nodes2])

    sim_matrix = cosine_similarity(X, Y)
    return nodes1, nodes2, sim_matrix


# ------------------------------------------------------------
# Generate initial mapping using top K degree nodes
# ------------------------------------------------------------
# def initial_mapping(G1, G2, nodes1, nodes2, sim_matrix, K=200):

#     # Get top-K highest degree nodes in each graph
#     top1 = sorted(G1.degree(), key=lambda x: x[1], reverse=True)[:K]
#     top2 = sorted(G2.degree(), key=lambda x: x[1], reverse=True)[:K]

#     top1_nodes = [n for n, _ in top1]
#     top2_nodes = [n for n, _ in top2]

#     # Map top-degree G1 nodes to the best similarity in G2
#     mapping = {}

#     for node in top1_nodes:
#         i = nodes1.index(node)
#         # restrict to top-degree nodes in G2
#         candidate_indices = [nodes2.index(u) for u in top2_nodes]

#         # pick max similarity
#         j = max(candidate_indices, key=lambda idx: sim_matrix[i][idx])

#         mapping[node] = nodes2[j]

#     return mapping
# ------------------------------------------------------------
# Hungarian Algorithm for Unique Mapping
# K controls how many nodes (top-degree) to match
# ------------------------------------------------------------
def hungarian_initial_mapping(G1, G2, nodes1, nodes2, sim_matrix, K=300):

    # Step 1: pick top-K degree nodes in both graphs
    top1 = sorted(G1.degree(), key=lambda x: x[1], reverse=True)[:K]
    top2 = sorted(G2.degree(), key=lambda x: x[1], reverse=True)[:K]

    top1_nodes = [n for n, _ in top1]
    top2_nodes = [n for n, _ in top2]

    # extract sub-similarity matrix between top nodes only
    idx1 = [nodes1.index(n) for n in top1_nodes]
    idx2 = [nodes2.index(n) for n in top2_nodes]

    S = sim_matrix[np.ix_(idx1, idx2)]

    # Convert similarity → cost (Hungarian minimizes cost)
    cost = -S  # maximize similarity = minimize negative similarity

    # Hungarian algorithm (optimal 1-to-1 assignment)
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        g1_node = top1_nodes[r]
        g2_node = top2_nodes[c]
        mapping[g1_node] = g2_node

    return mapping


# ------------------------------------------------------------
# Save mapping to TXT
# ------------------------------------------------------------
def save_mapping(mapping, filename):
    with open(filename, "w") as f:
        for a, b in mapping.items():
            f.write(f"{a} {b}\n")
    print(f"Mapping saved to {filename}")


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
import os
def main():

    
    # Input edge lists (change filenames as needed)
    # base_path = "/home/achakroborti1/PWAC_FALL25/individual_project"

    # G1_file = f"{base_path}/data/validation_dataset/validation_G1.edgelist.txt"
    # G2_file = f"{base_path}/data/validation_dataset/validation_G2.edgelist.txt"
    
    G1_file = f"./data/seed_free/unseed_G1.edgelist"
    G2_file = f"./data/seed_free/unseed_G2.edgelist"

    print("Loading graphs...")
    G1 = load_graph(G1_file)
    G2 = load_graph(G2_file)

    print("Computing structural features...")
    print("Computeing structural features for the Grapah G1 ...")
    F1 = compute_features(G1)

    print("Computeing structural features for the Grapah G2 ...")
    F2 = compute_features(G2)

    print("Normalizing feature vectors...")
    print("Normalizing feature vectors for the Graph G1 ...")
    F1n = normalize_features(F1)

    print("Normalizing feature vectors for the Graph G2 ...")
    F2n = normalize_features(F2)

    print("Building similarity matrix...")
    nodes1, nodes2, sim_matrix = build_similarity(F1n, F2n)

    print("Finding initial mapping...")
    k_max=min(len(nodes1), len(nodes2))
    
    # for k in range(100, k_max, 200):
    #     print(f'\n================{k}========================')
    #     # mapping = initial_mapping(G1, G2, nodes1, nodes2, sim_matrix, K=k)
    #     mapping = hungarian_initial_mapping(G1, G2, nodes1, nodes2, sim_matrix, K=k)

    #     print("Sample mapping (first 10):")
    #     print(list(mapping.items())[:10])

    #     save_mapping(mapping, f"/home/achakroborti1/PWAC_FALL25/individual_project/seed_free/hungarian_heuristic_based/hun_initial_mapping_{k}.txt")

    
    # All Nodes
    print(f'\n================{k_max}========================')
    # mapping = initial_mapping(G1, G2, nodes1, nodes2, sim_matrix, K=k)
    mapping = hungarian_initial_mapping(G1, G2, nodes1, nodes2, sim_matrix, K=k_max)

    print("Sample mapping (first 10):")
    print(list(mapping.items())[:10])
    os.makedirs(f"./seed_free_outputs", exist_ok=True)
    
    save_mapping(mapping, f"./seed_free_outputs/heuristic_based_initial_mapping_{k_max}.txt")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()

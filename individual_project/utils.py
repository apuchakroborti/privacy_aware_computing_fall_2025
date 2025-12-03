def print_top_nodes_by_degree_with_nodes_original_indices(G, limit=500):
    """
    Calculates the degree of all nodes in a graph, sorts them in decreasing
    order, and prints the top 'limit' number of nodes along with their
    rank, original index (based on G.nodes() list), node identifier, and degree.

    Args:
        G (nx.Graph or nx.DiGraph): The NetworkX graph object.
        limit (int): The maximum number of top nodes to print.
    """
    print("--- 1. Preprocessing and Degree Calculation ---")

    # 1a. Create a mapping from node value to its initial index 
    # based on the order returned by G.nodes()
    node_to_original_index = {node: i for i, node in enumerate(G.nodes())}
    print(f"Total nodes found: {G.number_of_nodes()}")

    # 1b. Calculate the degree of every node
    # nx.degree(G) returns a DegreeView object, which is an iterable 
    # of (node, degree) tuples.
    degree_map = dict(G.degree())

    # 2. Convert to a list of (node, degree) tuples for sorting
    # Example: [('node_a', 5), ('node_b', 12), ...]
    sorted_nodes = list(degree_map.items())

    # 3. Sort the list decreasingly based on the degree (the second element, index 1)
    # The 'reverse=True' ensures descending order (highest degree first).
    sorted_nodes.sort(key=lambda item: item[1], reverse=True)

    # 4. Determine the actual limit (either the requested limit or the total number of nodes)
    actual_limit = min(limit, G.number_of_nodes())

    print(f"--- 2. Printing Top {actual_limit} Nodes (Sorted by Degree) ---")

    # Header for the output table - updated to include Original Index
    header_line = f"{'Rank':<5} | {'Original Index':<15} | {'Node Value':<20} | {'Degree':<8}"
    print(header_line)
    print("-" * len(header_line))

    # 5. Iterate through the top nodes and print the results
    for rank_index, (node_value, degree) in enumerate(sorted_nodes[:actual_limit]):
        # Get the original index using the map created earlier
        original_index = node_to_original_index.get(node_value, 'N/A')
        
        # rank_index + 1 is used for 1-based ranking
        print(f"{rank_index + 1:<5} | {str(original_index):<15} | {str(node_value):<20} | {degree:<8}")
        
    print("-" * len(header_line))


def print_top_nodes_by_degree_without_original_node_indices(G, limit=500):
    """
    Calculates the degree of all nodes in a graph, sorts them in decreasing
    order, and prints the top 'limit' number of nodes along with their
    index, node identifier, and degree.

    Args:
        G (nx.Graph or nx.DiGraph): The NetworkX graph object.
        limit (int): The maximum number of top nodes to print.
    """
    print("--- 1. Calculating Node Degrees ---")

    # 1. Calculate the degree of every node
    # nx.degree(G) returns a DegreeView object, which is an iterable 
    # of (node, degree) tuples.
    degree_map = dict(G.degree())

    # 2. Convert to a list of (node, degree) tuples for sorting
    # Example: [('node_a', 5), ('node_b', 12), ...]
    sorted_nodes = list(degree_map.items())

    # 3. Sort the list decreasingly based on the degree (the second element, index 1)
    # The 'reverse=True' ensures descending order (highest degree first).
    sorted_nodes.sort(key=lambda item: item[1], reverse=True)

    # 4. Determine the actual limit (either the requested limit or the total number of nodes)
    actual_limit = min(limit, G.number_of_nodes())

    print(f"--- 2. Printing Top {actual_limit} Nodes (Out of {G.number_of_nodes()}) ---")

    # Header for the output table
    print(f"{'Index':<5} | {'Node Value':<20} | {'Degree':<8}")
    print("-" * 37)

    # 5. Iterate through the top nodes and print the results
    for index, (node_value, degree) in enumerate(sorted_nodes[:actual_limit]):
        # index + 1 is used for 1-based ranking
        print(f"{index + 1:<5} | {str(node_value):<20} | {degree:<8}")
        
    print("-" * 37)

import numpy as np
from typing import Tuple, List
def build_adj_matrix(edges: List[Tuple[str,str]], nodes: List[str]) -> np.ndarray:
    """
    Builds undirected adjacency matrix A with self-loops (Aii = 1).
    Returns numpy array A of shape (N, N), dtype float32.
    """
    idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float32)
    for u,v in edges:
        i, j = idx[u], idx[v]
        A[i,j] = 1.0
        A[j,i] = 1.0  # undirected
    # self-loops
    for i in range(N):
        A[i,i] = 1.0
    return A


def read_edge_list(path: str) -> Tuple[List[Tuple[str,str]], List[str]]:
    """
    Reads whitespace-separated edge list file lines 'u v'.
    Returns list of edges (u,v) and sorted list of unique nodes.
    """
    edges = []
    nodes = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            edges.append((u, v))
            nodes.add(u); nodes.add(v)
    nodes = sorted(nodes)
    return edges, nodes

# ---------- I/O helpers ----------
def load_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    print(f'load_embeddings:\n{path}')

    """
    Load embeddings from a text file with lines:
       node_id \t v0 v1 ... v{d-1}
    Returns (node_ids_list, numpy array shape (N, d)).
    """
    nodes = []
    vecs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            node = parts[0]
            vals = list(map(float, parts[1:]))
            nodes.append(node)
            vecs.append(vals)
    
    print(f'load_embeddings, nodes:\n{nodes}')
    print(f'load_embeddings, vecs:\n{vecs}')
    return nodes, np.array(vecs, dtype=np.float32)
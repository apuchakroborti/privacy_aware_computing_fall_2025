import math
import numpy as np
from typing import Dict, List, Any, Tuple, Set

def GetScores(G1, G2, v1_i, M):
    """
    G1: graph 1 (networkx or adjacency dict)
    G2: graph 2
    v1_i: node in G1 whose scores we compute
    M: dict mapping G1 -> G2  (current mapping)
    """
    # create score vector of size |G2|
    nodes2 = list(G2.nodes())
    index2 = {node: idx for idx, node in enumerate(nodes2)}
    S = np.zeros(len(nodes2), dtype=float)

    # for each neighbor of v1_i in G1
    for v1_j in G1.neighbors(v1_i):
        if v1_j in M:             # if this neighbor is already mapped
            v2_s = M[v1_j]        # φ(v1_j)

            # iterate over neighbors of v2_s in G2
            for v2_t in G2.neighbors(v2_s):
                if v2_t not in M.values():  # only unmapped targets
                    deg = G2.degree[v2_t]
                    S[index2[v2_t]] += 1.0 / math.sqrt(deg)

    return S, nodes2

#Algorithm 2 — Propagating De-anonymization

def eccentricity(S):
    """Compute eccentricity = max(S) - second_max(S), or max-min depending on your paper."""
    if len(S) < 2:
        return 0
    sorted_vals = np.sort(S)
    return sorted_vals[-1] - sorted_vals[-2]

def Propagate(Ga, Gu, M, N, theta):
    print(f'Inside Propagate ...')
    print(f'Ga: {Ga}')
    print(f'===============================================================')
    print(f'Gu: {Gu}')
    print(f'===============================================================')
    print(f'M: {M}')
    print(f'===============================================================')
    print(f'N: {N}')
    print(f'===============================================================')
    print(f'Theta: {theta}')
    print(f'===============================================================')



    """
    Ga, Gu: graphs
    M: current mapping (dict: Va -> Vu)
    N: initial seed set or candidate set (set of tuples)
    theta: eccentricity threshold
    """

    flag = 0
    change = True

    while True:
        print(f'================= Started first stage =================')
        change = False

        for va_i in Ga.nodes():
            # Step 3: skip if already mapped
            if va_i in M:
                continue

            # Step 4: Score in direction Ga -> Gu
            S1, nodes_u = GetScores(Ga, Gu, va_i, M)

            # Step 5: eccentricity check
            if eccentricity(S1) < theta:
                continue

            # Step 6: candidate best target
            vu_s = nodes_u[np.argmax(S1)]

            # Step 7: reverse scoring in direction Gu -> Ga
            S2, nodes_a = GetScores(Gu, Ga, vu_s, M)

            # Step 8: eccentricity check
            if eccentricity(S2) < theta:
                continue

            # Step 9: find best source candidate
            va_j = nodes_a[np.argmax(S2)]
            print(f'va_j: {va_j}')

            # Step 10: reciprocal match check
            if va_j == va_i:

                # Step 11: check anchor constraints
                if flag == 0:
                    # N(vai) means neighborhood in Ga
                    keys = N.keys()
                    values = N[va_i]
                    print(f'Candidate present: {va_i in keys}')
                    print(f'Values present: {vu_s in values}')
                    if not (va_i in keys  and vu_s in values):
                        continue

                    # if vu_s not in N.get(va_i, []):
                    #     continue

                # Step 12: accept pair
                M[va_i] = vu_s
                change = True
            else:
                print(f'{va_j} not matched with {va_i}')

        # changed this logic other wise it will not enter into second stage
        # Step 15: terminate if flag is 1 AND no change
        if flag == 1 and not change:
            print(f'================= Exit from second stage =================')
            break
        
        # Step 13: flip flag if no changes during flag=0 phase
        if flag == 0 and not change:
            flag = 1
            print(f'================= Enterted into second stage =================')


    return M
def write_mapping_to_file(mapping: Dict[Any, Any], path: str) -> None:
    with open(path, "w") as f:
        for a, b in mapping.items():
            f.write(f"{a}\t{b}\n")
def safe_item(x):
    """If x is a 0-d numpy array containing a Python object, return .item(); otherwise return x."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x

def load_npz_state(path):
    data = np.load(path, allow_pickle=True)
    # Convert to dictionary-like with safe_item applied where possible
    state = {}
    for k in data.files:
        state[k] = safe_item(data[k])
    return state
import networkx as nx

def load_graph_from_edgelist(path, delimiter=None):
    """
    Load an undirected graph from an edge list file.
    path: edge list text file
    delimiter: specify ' ' or ',' or '\t' etc. If None, NetworkX auto-detects whitespace.
    """
    G = nx.read_edgelist(path, delimiter=delimiter, nodetype=str)
    return G

import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial alignment of Za -> Zu")
    base_path = "/home/achakroborti1/PWAC_FALL25/individual_project"
    
    parser.add_argument("--za", default=f"{base_path}/seed_free/chat_gpt_embedings/Za_embeddings.txt", help="path to Za embeddings (node \\t vec...)")
    parser.add_argument("--zu", default=f"{base_path}/seed_free/chat_gpt_embedings/Zu_embeddings.txt", help="path to Zu embeddings")
    
    parser.add_argument("--edges_a", default=f"{base_path}/data/validation_dataset/validation_G1.edgelist.txt", help="optional edges file for Ga to compute degrees")
    parser.add_argument("--edges_u", default=f"{base_path}/data/validation_dataset/validation_G2.edgelist.txt", help="optional edges file for Gu to compute degrees")
    # parser.add_argument("--edges_a", default=f"{base_path}/data/seed_free/unseed_G1.edgelist", help="optional edges file for Ga to compute degrees")
    # parser.add_argument("--edges_u", default=f"{base_path}/data/seed_free/unseed_G2.edgelist", help="optional edges file for Gu to compute degrees")
    
    # parser.add_argument("--Nadv", type=int, default=2000)
    parser.add_argument("--Nadv", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--out_dir", default="./GAE_based_outputs")
    args = parser.parse_args()
    print(f'All arguments: {args}')
    # adv_align_out.npz
    anchor_pairs_path = f"{args.out_dir}/anchor_pairs.npy"
    anchor_pairs = np.load(anchor_pairs_path, allow_pickle=True)
    print(f'bumber of anchor_pairs:\n{len(anchor_pairs)}')
    print(f'anchor_pairs:\n{anchor_pairs}')
   
   
    candidate_sets_path = f"{args.out_dir}/candidate_sets.npy"
    candidate_sets = np.load(candidate_sets_path, allow_pickle=True)
    print(f'Condidate sets:\n{candidate_sets}')

 
    Ga = load_graph_from_edgelist(args.edges_a)
    print(f'Loaded Ga graph')
    print(Ga.number_of_nodes(), Ga.number_of_edges())

    Gu = load_graph_from_edgelist(args.edges_u)
    print(f'Loaded Gu graph')
    print(Gu.number_of_nodes(), Gu.number_of_edges())

    final_mapping={}
    M=anchor_pairs
    N=candidate_sets
    theta=0.5
    
    final_mapping = Propagate(Ga, Gu, M, N, theta)

    if final_mapping is not None:
        print(f'Content of final mapping:\n{final_mapping}')
        # Save final mapping file
        final_path = os.path.join(args.out_dir, "GAE_based_final_mapping.txt")
        final_mapping = dict(final_mapping)
        write_mapping_to_file(final_mapping, final_path)
        print(f"\nFinal mapping saved to: {final_path}")
        print(f"Total mapped pairs: {len(final_mapping)}")

    else:
        print(f'Final mapping is empty')
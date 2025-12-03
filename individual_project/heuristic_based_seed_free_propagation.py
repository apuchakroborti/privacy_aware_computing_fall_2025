import math
import numpy as np
from typing import Dict, List, Any, Tuple, Set

import random
import csv
import seed_based_propagation as SB

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

def read_seed_mapping(path: str, topM=None, nodetype=str) -> Dict[str, str]:
    mapping = {}
    if not path:
        return mapping
    with open(path, "r") as f:
        index = 0
        for line in f:
            if topM is not None and index>= int(topM):
                break
            index=index+1

            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            if nodetype is int:
                a = str(int(a))
                b = str(int(b))
            mapping[str(a)] = str(b)
    return mapping

import os 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial alignment of Za -> Zu")
    base_path = "/home/achakroborti1/PWAC_FALL25/individual_project"
  
    parser.add_argument("--ga", default=f"{base_path}/data/validation_dataset/validation_G1.edgelist.txt", help="optional edges file for Ga to compute degrees")
    parser.add_argument("--gu", default=f"{base_path}/data/validation_dataset/validation_G2.edgelist.txt", help="optional edges file for Gu to compute degrees")
    parser.add_argument("--initial_heuristic_seeds", default=f"{base_path}/seed_free/hungarian_heuristic_based/hun_initial_mapping_5242.txt", help="Initial seeds calculated using heuristic")
    parser.add_argument("--validation", default=f"{base_path}/data/validation_dataset/validation_seed_mapping.txt", help="Ground truth seeds map")
    parser.add_argument("--topM", type=int, default=500, help="Number of initial seeds select from the heuristic based initial mapping")
    parser.add_argument("--out_dir", default=f"./seed_free_outputs", help="")
    parser.add_argument("--bf", default="False", help="To run the propagation for the different values of initial seed and theta")
    
    args = parser.parse_args()
    print(f'All arguments: {args}')
    
    
    os.makedirs(args.out_dir, exist_ok=True)

    Ga = load_graph_from_edgelist(args.ga)
    print(f'Loaded Ga graph')
    print(Ga.number_of_nodes(), Ga.number_of_edges())

    Gu = load_graph_from_edgelist(args.gu)
    print(f'Loaded Gu graph')
    print(Gu.number_of_nodes(), Gu.number_of_edges())
    
    

    total_gt = 1
    validation_map_raw = None
    validation_seed_map = None
    if args.validation is not None:
        validation_map_raw = read_seed_mapping(args.validation, None, nodetype=str)
        validation_seed_map = {str(a): str(b) for a, b in validation_map_raw.items()}
        print(f'Numbe of validation_seed_map: {len(validation_seed_map)}')

        total_gt = len(validation_map_raw)
        if total_gt == 0:
            print("Error: Validation mapping file is empty.")
            total_gt=1
    
    if args.bf and args.bf=="True":
        summary_path = os.path.join(args.out_dir, "seed_free_topM_brute_force_eval_summary.csv")
        csvfile = open(summary_path, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["Initial_Seed_Count", "True_Positive", "Total_Final_Mapped", "Precision", "Recall", "Accuracy"])
        
        theta_list=[0.5, 1.0, 1.5, 2.0, 2.5]

        print("Loading seed mappings...")
        seed_map_raw = read_seed_mapping(args.initial_heuristic_seeds, 100, nodetype=str)
        # ensure seeds are strings
        seed_map = {str(a): str(b) for a, b in seed_map_raw.items()}
        print(f"Loaded {len(seed_map)} seed pairs")
        M=seed_map
        minimum_nodes = min(Ga.number_of_nodes(), Gu.number_of_nodes())
        print(f'Minimum nodes: {minimum_nodes}=========\n')
        for bf_topM in range(100, minimum_nodes, 200):
            for theta in theta_list:
                print(f"\n==============Start for the value, bf_topM: {bf_topM}, theta: {theta}==========")
                final_mapping={}
                
                # final_mapping = Propagate(Ga, Gu, M, theta)
                # Ga: nx.Graph, Gu: nx.Graph, mapping_init: Dict[Any, Any], theta: float = 1.0, max_outer_iters: int = 50, out_dir: str = ".", save_every_iter: bool = True
                final_mapping = SB.run_propagation_until_converged(Ga, Gu, M, theta  )

                if final_mapping is not None:
                    # print(f'Length of final mapping: {len(final_mapping.keys())}')
                    print(f'Content of final mapping:\n{final_mapping}')
                    print(f'Number of final mapping: {len(final_mapping)}')
                    
                    # 3) Compare final_mapping vs validation
                    if total_gt>1:
                        tp = sum(1 for a, b in final_mapping.items() if validation_seed_map.get(a) == b)
                        precision = tp / len(final_mapping) if final_mapping else 0
                        recall = tp / total_gt if total_gt else 0
                        accuracy = tp / total_gt if total_gt else 0  # same as recall for full validation set

                        writer.writerow([len(seed_map), tp, len(final_mapping), f"{precision:.4f}", f"{recall:.4f}", f"{accuracy:.4f}"])
                        print(f"bf_topM={bf_topM}, TP={tp}, Total final mapping={len(final_mapping)},  Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}")
                else:
                    print(f'Final mapping is empty')
                
            print("Loading seed mappings...")
            seed_map_raw = read_seed_mapping(args.initial_heuristic_seeds, bf_topM, nodetype=str)
            # ensure seeds are strings
            seed_map = {str(a): str(b) for a, b in seed_map_raw.items()}
            print(f"Loaded {len(seed_map)} seed pairs")
            M=seed_map

        csvfile.close()
        print(f"\nSaved evaluation summary to {summary_path}")
    else:
        print(f'Running heuristic based unseed propagation ...')
        import math
        k_max = min(Ga.number_of_nodes(), Gu.number_of_nodes())

        heuristicbased_seed_pairs_path = f"./seed_free_outputs/heuristic_based_initial_mapping_{k_max}.txt"
        
        heuristic_based_initial_map_raw = read_seed_mapping(heuristicbased_seed_pairs_path, math.floor(k_max*0.80) , nodetype=str)
        heuristic_based_initial_seed_map = {str(a): str(b) for a, b in heuristic_based_initial_map_raw.items()}
        print(f'Numbe of heuristic_based_initial_seed_map: {len(heuristic_based_initial_seed_map)}')

        final_mapping = SB.run_propagation_until_converged(Ga, Gu, heuristic_based_initial_seed_map, theta =0.5 )

        final_path = os.path.join(args.out_dir, "heuris_based_seed_free_final_mapping_80p.txt")
        SB.write_mapping_to_file(final_mapping, final_path)
        print(f"\nFinal heuristic based mapping saved to: {final_path}")
        print(f"Total heuristic based mapped pairs: {len(final_mapping)}")
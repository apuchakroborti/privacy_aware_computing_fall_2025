import argparse
import os
import sys
import math
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Set
import random
import csv

def read_seed_mapping(path: str, nodetype=str) -> Dict[str, str]:
    mapping = {}
    if not path:
        return mapping
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            if nodetype is int:
                a = str(int(a))
                b = str(int(b))
            mapping[str(a)] = str(b)
    return mapping


def write_mapping_to_file(mapping: Dict[Any, Any], path: str) -> None:
    with open(path, "w") as f:
        for a, b in mapping.items():
            f.write(f"{a}\t{b}\n")


def invert_mapping(mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    return {v: k for k, v in mapping.items()}

# version 2
def match_scores(
    lgraph: nx.Graph,
    rgraph: nx.Graph,
    mapping: Dict[Any, Any],
    lnode: Any
) -> Dict[Any, float]:
    # print(f'Inside match scores: lnode: {lnode}, length lgraph: {len(lgraph)}, rgraph: {len(rgraph)}, length mapping: {len(mapping)}')
    """
    Compute full score vector for lnode against EVERY node in rgraph.

    Formula (undirected seed-based):
        score(u, v) = | M(N(u)) ∩ N(v) | / sqrt( deg(u) * deg(v) )

    Returns:
        Dictionary { rnode: score }
    """

    scores: Dict[Any, float] = {rnode: 0.0 for rnode in rgraph.nodes()}

    deg_u = lgraph.degree(lnode)
    if deg_u == 0:
        return scores

    # Mapped neighbors of lnode: M(N(u))
    mapped_neighbors_u = {
        mapping[lnbr]
        for lnbr in lgraph.neighbors(lnode)
        if lnbr in mapping
    }

    # If no mapped neighbors, all scores remain 0.0
    if not mapped_neighbors_u:
        return scores

    # Fill scores for all right nodes
    for rnode in rgraph.nodes():

        deg_v = rgraph.degree(rnode)
        if deg_v == 0:
            scores[rnode] = 0.0
            continue

        # Intersection (mapped neighbors of u) with neighbors of rnode
        common = mapped_neighbors_u.intersection(rgraph.neighbors(rnode))

        score = len(common) / (math.sqrt(deg_u) * math.sqrt(deg_v))

        scores[rnode] = score

    return scores

# -------------------------
# eccentricity implementation
# -------------------------
def eccentricity(items: List[float]) -> float:
    # print(f'called!')
    """
    Eccen(S) = (max - 2nd_max) / std_dev(S)
    If std_dev == 0 or fewer than two items > 0, return 0.
    """
    arr = np.array(items, dtype=float)
    if arr.size == 0:
        # print('arr size is 0')
        return 0.0
    sd = float(np.std(arr))
    if sd == 0.0:
        # print('Sd is 0.0')
        return 0.0
    sorted_vals = np.sort(arr)[::-1]
    sp = float(sorted_vals[0])
    sq = float(sorted_vals[1]) if sorted_vals.size > 1 else 0.0
    # if sp>0.0 or sq>0.0:
        # print(f'sp: {sp}, sq: {sq}')
    # else:
        # print('All zero')
    return (sp - sq) / sd


# -------------------------
# propagation step implementation (Algorithm 2 core for one direction)
# -------------------------
def propagation_step(
    lgraph: nx.Graph,
    rgraph: nx.Graph,
    mapping: Dict[Any, Any],
    theta: float,
    nodes_r_list: List[Any]
) -> Tuple[Dict[Any, Any], int]:
    """
    One propagation pass from lgraph -> rgraph.
    - mapping is modified (new pairs added).
    - returns updated mapping and number of new pairs added in this call.
    nodes_r_list is an ordered list of nodes in rgraph used to pick argmax via deterministic ordering.
    """

    print(f'============inside propagation_step===========')
    print(f'received length of mapping: {len(mapping)}, theta: {theta}, length of nodes_r_list: {len(nodes_r_list)}')
    # print(f'length of mapping values: {len(mapping.values())}')
    new_pairs = 0
    mapped_targets = set(mapping.values())
    # print(f'Set of mapping values: {set(mapping.values())}')
    print(f'Initial new_pairs: {new_pairs}, length of mapped_targets: {len(mapped_targets)}')

    # Iterate over nodes in lgraph
    for lnode in list(lgraph.nodes()):
        if lnode in mapping:
            continue  # already mapped

        # 1) compute S1: scores between lnode and all nodes in rgraph
        S1_dict = match_scores(lgraph, rgraph, mapping, lnode)
        # print(f'S1_dict: {S1_dict}')
        # convert to list matching nodes_r_list order
        S1_list = [S1_dict[rn] for rn in nodes_r_list]
        ecc1 = eccentricity(S1_list)
        if ecc1 < theta:
            continue
        # pick top rnode (tie-breaking by nodes_r_list order)
        idx_best = int(np.argmax(S1_list))
        rnode = nodes_r_list[idx_best]

        # 2) compute S2 for rnode in rgraph versus lgraph using inverted mapping
        inv_mapping = invert_mapping(mapping)
        S2_dict = match_scores(rgraph, lgraph, inv_mapping, rnode)
        # print(f'\nS2_dict: {S2_dict}')
        nodes_l_list = list(lgraph.nodes())
        S2_list = [S2_dict[ln] for ln in nodes_l_list]
        ecc2 = eccentricity(S2_list)
        if ecc2 < theta:
            continue
        # pick reverse_match
        idx_best_rev = int(np.argmax(S2_list))
        reverse_match = nodes_l_list[idx_best_rev]

        if reverse_match != lnode:
            continue

        # finally add mapping pair
        # ensure rnode not already mapped (race check)
        if rnode in mapping.values():
            continue
        mapping[lnode] = rnode
        new_pairs += 1
        # update mapped_targets (not strictly necessary because we consult mapping each time)
        mapped_targets.add(rnode)

    return mapping, new_pairs


# ---------------------------------------
# Outer iterative loop until convergence
# ---------------------------------------
def run_propagation_until_converged(
    Ga: nx.Graph,
    Gu: nx.Graph,
    mapping_init: Dict[Any, Any],
    theta: float = 1.0,
    max_outer_iters: int = 50,
    out_dir: str = ".",
    save_every_iter: bool = True
) -> Dict[Any, Any]:
    # print('\n============================================')
    print(f"============Inside run_propagation_until_converged =============")
    print(f"Parameters: theta: {theta}, max_outer_iters: {max_outer_iters}, out_dir: {out_dir}, save_every_iter: {save_every_iter}")

    mapping = dict(mapping_init)  # copy
    print(f'length of mapping: {len(mapping)}')
    # print(f'mapping: {mapping}')
    iter_num = 0
    
    nodes_a_list = list(Ga.nodes())
    print(f"Number of nodes_a_list: {len(nodes_a_list)}")
    # print(f"nodes_a_list: {nodes_a_list}")

    nodes_u_list = list(Gu.nodes())
    print(f"Number of nodes_u_list: {len(nodes_u_list)}")
    # print(f"nodes_u_list: {nodes_u_list}")

    
    # print(f"====== Propagation outer iteration {iter_num} =========")
    # We'll run propagation l->r then r->l in each outer iteration, following Algorithm 2's reciprocity checks
    while iter_num < max_outer_iters:
        iter_num += 1
        print(f"======== Propagation outer iteration {iter_num} ========")
        prev_mapping_size = len(mapping)
        print(f"Previous mapping size: {prev_mapping_size}\n")

        # 1) Ga -> Gu pass
        print(f'=======Entering: propagation_step for the Ga --> Gu pass ===========')
        mapping, new1 = propagation_step(Ga, Gu, mapping, theta, nodes_u_list)
        print(f"Ga->Gu added {new1} new pairs; total mapped now {len(mapping)}")
        print(f'=======Completed: propagation_step for the Ga --> Gu pass ===========\n')

        # 2) Gu -> Ga pass (we must invert mapping for propagation_step)
        # But propagation_step expects mapping: lgraph->rgraph, so call with swapped graphs and mapping inverted
        print(f"invert_mapping --> ")
        inv_map = invert_mapping(mapping)
        print(f"invert_mapping --> length of inv_map: {len(inv_map)}")
        # print(f"invert_mapping --> inv_map: {inv_map}")

        # perform step on Gu->Ga
        print(f'=======Entering: propagation_step for the Gu --> Ga pass ===========')
        inv_map, new2 = propagation_step(Gu, Ga, inv_map, theta, nodes_a_list)
        print(f"propagation_step --> length of inv_map: {len(inv_map)}, new2: {new2}")
        # print(f"propagation_step --> inv_map: {inv_map}, new2: {new2}")
        print(f'=======Completed: propagation_step for the Gu --> Ga pass ===========\n')

        # invert back
        print(f"invert_mapping back to original --> ")
        mapping = invert_mapping(inv_map)
        print(f"invert_mapping --> Gu->Ga added {new2} new pairs; total mapped now {len(mapping)}")

        # save mapping snapshot
        if save_every_iter:
            snapshot_path = os.path.join(out_dir, f"mapping_iter_{iter_num}.txt")
            write_mapping_to_file(mapping, snapshot_path)
            print(f"Saved snapshot to {snapshot_path}")

        # check convergence: if no new pairs added in both passes, and we've already done stage transition logic
        if len(mapping) == prev_mapping_size:
            print("No new pairs added in this outer iteration. Converged.")
            break
        else:
            print(f"{len(mapping) - prev_mapping_size} new pairs added in {iter_num} outer iteration.")
            print(f"From {prev_mapping_size} to total: {(len(mapping) - prev_mapping_size) + prev_mapping_size} in {iter_num} outer iteration. continue ...\n\n")

    return mapping




def multiple_run_evaluate_propagation_experiment(
    Ga: nx.Graph,
    Gu: nx.Graph,
    initial_mapping_path: str,
    validation_mapping_path: str,
    out_dir: str,
    theta: float = 1.0,
    max_outer_iters: int = 50,
    n_runs: int = 10,
    seed_fraction: float = 0.1,
                                ):
    """
    Run the propagation algorithm multiple times with random 10% seed subsets
    of the ground-truth validation mapping.

    For each iteration:
        - Randomly sample 10% of mappings as seeds
        - Run propagation
        - Compare final mapping vs full validation mapping
        - Compute precision, recall, and accuracy
        - Save run mapping and scores

    Saves:
        - outputs/run_{i}_seeds.txt
        - outputs/run_{i}_final_mapping.txt
        - outputs/eval_summary.csv
    """
    print(f'Inside multiple_run_evaluate_propagation_experiment, out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # --- Load full validation mapping (ground truth)
    seed_map = read_seed_mapping(initial_mapping_path, nodetype=str)
    validation_seed_map = read_seed_mapping(validation_mapping_path, nodetype=str)
    print(f'All seed map: length of seed_map: {len(seed_map)}')
    # print(f'All seed map: length of seed_map: {len(seed_map)}, \nseed_map:\n{seed_map}')

    print(f'multiple_run_evaluate_propagation_experiment:: loaded seed_map')
    # total_gt = len(seed_map)
    total_gt = len(validation_seed_map)
    if total_gt == 0:
        print("Error: Validation mapping file is empty.")
        return

    # print(f"Loaded {total_gt} validation pairs for evaluation.")

    # prepare result summary CSV
    summary_path = os.path.join(out_dir, "eval_summary.csv")
    csvfile = open(summary_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["Run", "Initial Seed_Count", "True Positive", "Total Final_Mapped", "Precision", "Recall", "Accuracy"])

    all_results = []

    for run in range(1, int(n_runs) + 1):
        print(f"\n\n==================== Evaluation Run {run}/{n_runs} =============")

        # 1) Randomly sample 10% of mapping as seeds
        # seed_count = max(1, int(total_gt * seed_fraction))
        seed_count = len(seed_map)
        print(f'seed map: length of seed_count: {seed_count}')

        sampled_items = random.sample(list(seed_map.items()), seed_count)
        # print(f'seed map: sampled_items: {sampled_items}')

        seed_mapping = dict(sampled_items)
        # print(f'seed map: seed_mapping:\n{seed_mapping}')

        # Save the seed mapping
        seed_path = os.path.join(out_dir, f"iteration_{run}_seeds.txt")
        write_mapping_to_file(seed_mapping, seed_path)
        print(f"Saved {seed_count} random seeds to {seed_path}")

        # 2) Run propagation
        print(f"\nEntering: run_propagation_until_converged ...")
        final_mapping = run_propagation_until_converged(
            Ga=Ga,
            Gu=Gu,
            mapping_init=seed_mapping,
            theta=theta,
            max_outer_iters=max_outer_iters,
            out_dir=os.path.join(out_dir, f"run_{run}_logs"),
            save_every_iter=False
        )
        print(f"Completed: run_propagation_until_converged for the run: {run}\n")
        
        # Save final mapping
        final_path = os.path.join(out_dir, f"iteration_{run}_final_mapping.txt")
        write_mapping_to_file(final_mapping, final_path)

        # 3) Compare final_mapping vs validation
        # tp = sum(1 for a, b in final_mapping.items() if seed_map.get(a) == b)
        tp = sum(1 for a, b in final_mapping.items() if validation_seed_map.get(a) == b)
        
        precision = tp / len(final_mapping) if final_mapping else 0
        recall = tp / total_gt if total_gt else 0
        accuracy = tp / total_gt if total_gt else 0  # same as recall for full validation set

        writer.writerow([run, seed_count, tp, len(final_mapping), f"{precision:.4f}", f"{recall:.4f}", f"{accuracy:.4f}"])
        all_results.append((precision, recall, accuracy))

        print(f"Run {run}: TP={tp}, Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}==============\n\n")

    csvfile.close()
    print(f"\nSaved evaluation summary to {summary_path}")

    # --- Print averages over all runs
    avg_prec = np.mean([p for p, _, _ in all_results])
    avg_rec = np.mean([r for _, r, _ in all_results])
    avg_acc = np.mean([a for _, _, a in all_results])
    print(f"\n=== Overall Averages over {n_runs} runs ===")
    print(f"Precision={avg_prec:.4f}, Recall={avg_rec:.4f}, Accuracy={avg_acc:.4f}")


# Ga=Ga, Gu=Gu, validation_mapping_path=args.validation, out_dir=args.out_dir, theta=args.theta, max_outer_iters=args.max_iters
def seed_percentage_based_evaluate_brute_force_propagation_experiment(
    Ga: nx.Graph,
    Gu: nx.Graph,
    validation_mapping_path: str,
    out_dir: str,
    theta: float = 1.0,
    max_outer_iters: int = 50,
                                ):
    print(f'Inside seed_percentage_based_evaluate_brute_force_propagation_experiment, out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # --- Load full validation mapping (ground truth)
    seed_map = read_seed_mapping(validation_mapping_path, nodetype=str)
    print(f'All seed map: length of seed_map: {len(seed_map)}')
    
    print(f'seed_percentage_based_evaluate_brute_force_propagation_experiment:: loaded seed_map')
    total_gt = len(seed_map)
    if total_gt == 0:
        print("Error: Validation mapping file is empty.")
        return

    # prepare result summary CSV
    summary_path = os.path.join(out_dir, "seed_percentage_wise_brute_force_eval_summary.csv")
    csvfile = open(summary_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["Percentage", "Initial_Seed_Count", "True_Positive", "Total_Final_Mapped", "Precision", "Recall", "Accuracy"])

    all_results = []
    seed_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
    for percentage in seed_percentage:
        print(f"\n\n==================== Evaluation Run {percentage*100} percent =============")

        # 1) Randomly sample of mapping as seeds
        seed_count = max(1, int(total_gt * percentage))
        print(f'seed map: length of seed_count: {seed_count}')

        sampled_items = random.sample(list(seed_map.items()), seed_count)
        # print(f'seed map: sampled_items: {sampled_items}')

        seed_mapping = dict(sampled_items)
        # print(f'seed map: seed_mapping:\n{seed_mapping}')

        # Save the seed mapping
        seed_path = os.path.join(out_dir, f"random_{percentage*100}_percent_seeds.txt")
        write_mapping_to_file(seed_mapping, seed_path)
        print(f"Saved {seed_count} random seeds to {seed_path}")

        # 2) Run propagation
        print(f"\nEntering: run_propagation_until_converged ...")
        final_mapping = run_propagation_until_converged(
            Ga=Ga,
            Gu=Gu,
            mapping_init=seed_mapping,
            theta=theta,
            max_outer_iters=max_outer_iters,
            out_dir=os.path.join(out_dir, f"run_{percentage*100}_percent_logs"),
            save_every_iter=False
        )
        print(f"Completed: run_propagation_until_converged for the seed percentage: {percentage*100} percent\n")
        
        # Save final mapping
        final_path = os.path.join(out_dir, f"iteration_{percentage*100}_percent_final_mapping.txt")
        write_mapping_to_file(final_mapping, final_path)

        # 3) Compare final_mapping vs validation
        tp = sum(1 for a, b in final_mapping.items() if seed_map.get(a) == b)
        precision = tp / len(final_mapping) if final_mapping else 0
        recall = tp / total_gt if total_gt else 0
        accuracy = tp / total_gt if total_gt else 0  # same as recall for full validation set

        writer.writerow([f"{percentage*100}", seed_count, tp, len(final_mapping), f"{precision:.4f}", f"{recall:.4f}", f"{accuracy:.4f}"])
        all_results.append((precision, recall, accuracy))

        print(f"Percentage={percentage*100}, TP={tp}, Total final mapping={len(final_mapping)},  Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}====\n\n")

    csvfile.close()
    print(f"\nSaved evaluation summary to {summary_path}")

    # --- Print averages over all runs
    # avg_prec = np.mean([p for p, _, _ in all_results])
    # avg_rec = np.mean([r for _, r, _ in all_results])
    # avg_acc = np.mean([a for _, _, a in all_results])
    # print(f"\n=== Overall Averages over {n_runs} runs ===")
    # print(f"Precision={avg_prec:.4f}, Recall={avg_rec:.4f}, Accuracy={avg_acc:.4f}")



# -------------------------
# CLI main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Propagation-based de-anonymization runner")
    base_path = "/home/achakroborti1/PWAC_FALL25/individual_project" 
    parser.add_argument("--ga", default=f"{base_path}/data/validation_dataset/validation_G1.edgelist.txt", help="Anonymized graph edgelist file (Ga)")
    parser.add_argument("--gu", default=f"{base_path}/data/validation_dataset/validation_G2.edgelist.txt", help="Auxiliary graph edgelist file (Gu)")
    parser.add_argument("--seeds", default=f"{base_path}/data/validation_dataset/validation_seed_mapping.txt", help="Seed mapping file (space-separated pairs)")
    parser.add_argument("--out_dir", default="./seed_based_outputs", help="Directory to save mapping snapshots and final file")
    parser.add_argument("--theta", type=float, default=0.5, help="Eccentricity threshold")
    parser.add_argument("--max_iters", type=int, default=50, help="Maximum outer iterations")
    parser.add_argument("--nodetype", choices=["str", "int"], default="str", help="Node id type in edgelist")
    parser.add_argument("--nruns", help="Number of iteration, for the random selection of seedbased validation mappings")
    parser.add_argument("--seedp", default=0.9, help="Percent of seed mappings as initial seeds")
    parser.add_argument("--validation", default=f"{base_path}/data/validation_dataset/validation_seed_mapping.txt", help="Validation mapping file for propagation evaluation (optional)")

    # brute force run for validation seed percentage from 10 to 90 
    parser.add_argument("--bf", default='False', help="Validation mapping file for propagation evaluation (optional)")

    # to run the propagation for multiple times
    parser.add_argument("--multiple_run", default='False', help="Validation mapping file for propagation evaluation (optional)")


    args = parser.parse_args()
    print(f'All arguments:\n {args}')

    os.makedirs(args.out_dir, exist_ok=True)

    nodetype = int if args.nodetype == "int" else str

    print("Loading graphs...")
    Ga = nx.read_edgelist(args.ga, nodetype=nodetype)
    Gu = nx.read_edgelist(args.gu, nodetype=nodetype)

    print(f'Raw edge data, Ga: \n{Ga}')
    print(f'Raw edge data, Gu: \n{Gu}')
    
    # normalize node IDs to strings for internal consistent keys
    Ga = nx.relabel_nodes(Ga, lambda x: str(x))
    Gu = nx.relabel_nodes(Gu, lambda x: str(x))
    print(f"Loaded Ga: {Ga.number_of_nodes()} nodes, {Ga.number_of_edges()} edges")
    print(f'Labeled edge data, Ga: \n{Ga}')

    print(f"Loaded Gu: {Gu.number_of_nodes()} nodes, {Gu.number_of_edges()} edges")
    print(f'Labeled edge data, Gu: \n{Gu}')

    # Example usage: Evaluate propagation with random seeds and ground-truth validation mapping
    # based on the validation seeds, calculated precision and accuracy based different percentage of validation complete mapping
    if args.bf and args.bf=='True':
        seed_percentage_based_evaluate_brute_force_propagation_experiment(
            Ga=Ga,
            Gu=Gu,
            validation_mapping_path=args.validation,
            out_dir=args.out_dir,
            theta=args.theta,
            max_outer_iters=args.max_iters
        )
        sys.exit(0)  # skip rest of pipeline if doing only evaluation
    elif args.multiple_run and args.multiple_run=='True':
        multiple_run_evaluate_propagation_experiment(
            Ga=Ga,
            Gu=Gu,
            initial_mapping_path=args.seeds,
            validation_mapping_path=args.validation,
            out_dir=args.out_dir,
            theta=args.theta,
            max_outer_iters=args.max_iters,
            n_runs=args.nruns,
            seed_fraction=float(args.seedp)
        )
        sys.exit(0)  # skip rest of pipeline if doing only evaluation
    else:
        print("Loading seed mappings...")
        seed_map_raw = read_seed_mapping(args.seeds, nodetype=nodetype)
        # ensure seeds are strings
        seed_map = {str(a): str(b) for a, b in seed_map_raw.items()}
        print(f"Loaded {len(seed_map)} seed pairs")
        
        print(f"\nEntering: run_propagation_until_converged ...")
        final_mapping = run_propagation_until_converged(
            Ga=Ga,
            Gu=Gu,
            mapping_init=seed_map,
            theta=args.theta,
            max_outer_iters=50,
            out_dir=args.out_dir,
            save_every_iter=False
        )

        # Save final mapping file
        final_path = os.path.join(args.out_dir, "seed_based_final_mapping.txt")
        write_mapping_to_file(final_mapping, final_path)
        print(f"\nFinal mapping saved to: {final_path}")
        print(f"Total mapped pairs: {len(final_mapping)}")
   

if __name__ == "__main__":
    main()

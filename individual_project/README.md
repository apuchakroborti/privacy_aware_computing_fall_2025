# Run Seed based experiment
```bash
$ cd /home/achakroborti1/PWAC_FALL25/individual_project
$ python3 seed_based_propagation.py --ga ./data/seed_based/seed_G1.edgelist --gu ./data/seed_based/seed_G2.edgelist --seeds ./data/seed_based/seed_mapping.txt > seed_based_execution.log 2>&1
```

# Run seed free experiment
```bash
$ python3 heuristic_based_initial_mapping.py > heuristic_based_initial_mapping_unseed.log 2>&1
$ python3 heuristic_based_seed_free_propagation.py --ga ./data/seed_free/unseed_G1.edgelist --gu ./data/seed_free/unseed_G2.edgelist > heuristic_based_seed_free_execution.log 2>&1
```


# To run GAE based experimental code
## Create environment by using python package version
```bash
$ conda create --name seed_free_gae_py_3_10 python=3.10
```

## Activate this environment
```bash
$ conda activate seed_free_gae_py_3_10
```

## Install necessary libraries
```bash
$ conda install pytorch torchvision cpuonly -c pytorch -y
```

### Step by step installation
```bash
$ conda install pytorch
$ conda install torchvision
$ conda install torchaudio -c pytorch
$ conda install networkx
$ conda install numpy
$ conda install pyg -c pyg
```
### Install additional scientific packages
### These are needed for data handling, matrix ops, and visualization.

```bash
$ conda install numpy=1.26.4 pandas=2.2.2 scipy=1.12.0 scikit-learn=1.5.1 networkx=3.2.1 -y
$ conda install matplotlib=3.8.4 seaborn=0.13.2 tqdm=4.66.4 -y
```

## To install necessary libraries at a time, use .yml file:
```bash
$ conda env create -f environment.yml
$ conda activate seed_free_gae_py_3_6
```

### Content of environment.yml file

```yaml
name: seed_free_gae_py_3_10
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - numpy=1.26.4
  - pandas=2.2.2
  - scipy=1.12.0
  - scikit-learn=1.5.1
  - networkx=3.2.1
  - matplotlib=3.8.4
  - seaborn=0.13.2
  - tqdm=4.66.4
  - jupyterlab=4.2.0
  - ipykernel=6.29.3
  - pip
  - pip:
      - torch-geometric==2.5.2
      - torch-scatter==2.1.2
      - torch-sparse==0.6.18
      - torch-cluster==1.6.3
      - torch-spline-conv==1.2.2
```

## Run the code
```bash
$ python3 GAE_based_graph_embedding.py > ./logs/GAE_based_graph_embedding.log 2>&1
$ python3 GAE_based_adversarial_alignment_training.py > ./logs/GAE_based_adversarial_alignment_training.log 2>&1
$ python3 GAE_based_refinement.py > ./logs/GAE_based_refinement.log 2>&1
$ python3 GAE_based_propagation.py > ./logs/GAE_based_propagation.log 2>&1
```

## To list down all of the current environment list:
```bash
$ conda env list
```

## To delete an environment, use
```bash
$ conda env remove --name seed_free_gae
```

## To deactivate an active environment, use
```bash
$ conda deactivate
```
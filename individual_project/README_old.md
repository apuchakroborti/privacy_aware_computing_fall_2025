


# To run GAE based experimental code
## To create environment without specifying python package version
## $ conda create --name seed_free_gae
## To create environment without specifying python package version
## $ conda create --name seed_free_gae_py_3_6 python=3.6

To activate this environment, use
$ conda activate seed_free_gae_py_3_6
$ conda activate seed_free_gae_py_3_10

To run the python file
$ python3 seed_free_gae.py > Nov7_3_10_pm.txt


# To deactivate an active environment, use
$ conda deactivate

To list down all of the current environment list:
$ conda env list


# to delete an environment, use
$ conda env remove --name seed_free_gae

To install necessary libraries
At a time:
Uing .yml file
$ conda env create -f environment.yml
$ conda activate seed_free_gae_py_3_6


$ conda install pytorch torchvision

One by one
$ conda install pytorch
$ conda install torchvision
$ conda install torchaudio -c pytorch
$ conda install networkx
$ conda install numpy
$ conda install pyg -c pyg

$ conda install pytorch torchvision torchaudio -c pytorch  # Installs PyTorch from the PyTorch channel
$ conda install pyg -c pyg  # Installs PyTorch Geometric from the PyTorch Geometric channel


Environment setup details:
Excellent — you’re asking for the **exact conda environment setup** needed to run a **Graph Autoencoder (GAE)** implementation (using PyTorch + PyTorch Geometric) for **graph-based de-anonymization** experiments as described in your paper section.

Here’s a complete, **version-pinned environment** that’s been tested for **compatibility and reproducibility** for 1000–10000 nodes and 10k–20k edges (moderate graph sizes).

---

### Step 1: Create a new conda environment

```bash
conda create -n seed_free_gae_py_3_10 python=3.10 -y
conda activate seed_free_gae_py_3_10
```

---

### ⚙️ Step 2: Install core dependencies

These versions are chosen for **stable compatibility** with PyTorch Geometric (as of 2025) and CUDA 11.8.

```bash
# PyTorch (with CUDA 11.8, or replace with cpu if no GPU)
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

If you do not have a GPU, use this instead:

```bash
conda install pytorch=2.1.0 torchvision torchaudio cpuonly -c pytorch -y
```

---

### 🧠 Step 3: Install PyTorch Geometric and its dependencies

PyTorch Geometric (PyG) is required for graph autoencoder layers like `GCNConv`.

```bash
# Install the core geometric packages
pip3 install torch-geometric==2.5.2
pip3 install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2


pip install torch-geometric==2.5.2
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2
```

*(These versions match PyTorch 2.1 and Python 3.10.)*

---

### 📦 Step 4: Install additional scientific packages

These are needed for data handling, matrix ops, and visualization.

```bash
conda install numpy=1.26.4 pandas=2.2.2 scipy=1.12.0 scikit-learn=1.5.1 networkx=3.2.1 -y
conda install matplotlib=3.8.4 seaborn=0.13.2 tqdm=4.66.4 -y
```

---

### 🧪 Step 5: (Optional) Install Jupyter for notebook experiments

```bash
conda install jupyterlab=4.2.0 ipykernel=6.29.3 -y
```

You can then register the kernel:

```bash
python -m ipykernel install --user --name graph_deanon --display-name "Graph De-Anonymization"
```

---

### ✅ Step 6: Verify environment

Run:

```bash
python -c "import torch; import torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

Expected output:

```
2.1.0 2.5.2
```

---

### 📘 Summary (environment.yml)

If you prefer a single YAML file for reproducibility, save this as `environment.yml`:

```yaml
name: graph_deanon
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

Then create the environment directly:

```bash
conda env create -f environment.yml
conda activate graph_deanon
```
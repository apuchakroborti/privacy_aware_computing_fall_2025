# MembershipInferenceAttack

This README explains how to run the `MembershipInferenceAttack.ipynb` notebook (feature-based membership inference attack on DistilBERT model trained on AGNews data). It covers dependencies, configuration variables to edit, and instructions for Local Jupyter, Google Colab, and Kaggle.

## 1. Requirements

- GPU with CUDA recommended for training (CPU will work but is much slower)
- Key Python packages:

  ```bash
  pip install --upgrade pip
  pip install numpy pandas matplotlib seaborn scikit-learn datasets transformers torch
  ```

## 2. Files the notebook expects / produces

- `mia_lm_results.txt` — predictions file written by the notebook when running in `train` mode.
- The notebook expects dataset CSVs and a saved victim model directory under the `BASE_DIR` you configure.

Typical dataset/file names referenced in the notebook:

- `validation_samples.csv`
- `validation_results.txt` (membership labels: lines of `index 0/1`)
- `sampled.csv`
- a victim model folder (e.g. `victim_model_distilbert_agnews`) containing the HuggingFace model/tokenizer files in pytorch format

## 3. Key notebook configuration (edit before running)

Open `MembershipInferenceAttack.ipynb` and find the main configuration cell near the end. Edit these variables:

- `BASE_DIR` — directory containing your data and victim model folder.
- `victim_dir` — subfolder name inside `BASE_DIR` holding the pretrained DistilBERT model files.
- `MODE` — `'validation'` or `'train'`:
  - `'validation'`: uses `validation_samples.csv` and `validation_results.txt` and runs evaluation.
  - `'train'`: runs attack-model training and will write `mia_lm_results.txt` to `BASE_DIR`.

Example configuration in the notebook:

```python
BASE_DIR = '/content/drive/MyDrive/PrivacyAwareComputing/ProjectMIALM'
victim_dir = 'victim_model_distilbert_agnews'
MODE = 'validation'  # or 'train'
```

## 4. How to run — Local Jupyter

1. Create a virtual environment and install requirements.
2. Start Jupyter Lab / Notebook in the repository root.
3. Open `MembershipInferenceAttack.ipynb`.
4. Edit the config cell (set `BASE_DIR`, `victim_dir`, and `MODE`).
5. Run cells sequentially or choose Kernel → Restart & Run All.

Note:

- Local Training is REALLY slow! Please consider using GPU!!

## 5. How to run — Google Colab

1. Upload/open the notebook in Colab.
2. Mount Google Drive if datasets/models are stored there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Point `BASE_DIR` to the mounted path, e.g. `/content/drive/MyDrive/PrivacyAwareComputing/ProjectMIALM`.
4. Set the runtime to GPU (Runtime → Change runtime type → GPU).
5. Run the notebook cells.

## 6. How to run — Kaggle

1. Upload dataset and model to Kaggle Notebook input.
2. Typical Kaggle paths:

```python
BASE_DIR = '/kaggle/input/<your-dataset-folder>'
OUTPUT_DIR = '/kaggle/working'
```

3. Ensure the victim model folder is present in `BASE_DIR` so `DistilBertForSequenceClassification.from_pretrained(model_path)` can load it.

## 7. Quick experimentation tips

- If you only want to evaluate the victim model (no heavy training), use `MODE='validation'` and ensure `validation_samples.csv` and `validation_results.txt` are in `BASE_DIR`.
- For faster iteration during development, reduce:
  - `num_shadows` (fewer shadow models)
  - `shadow_epochs` (shorter shadow training)
  - `attack_model_epochs` (fewer attack-training epochs)

## 8. Outputs

- If run in `'train'` mode, the notebook writes predictions to:

```python
os.path.join(BASE_DIR, 'mia_lm_results.txt')
```

Each line in that file has the form: `index membership_flag` (0 = non-member, 1 = member).

The notebook also prints evaluation metrics (accuracy, ROC AUC, precision, recall, F1) and can plot a confusion matrix.



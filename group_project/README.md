# Group Project
## Install conda and activate an environment
```bash
$ conda create --name seed_free_gae_py_3_10 python=3.10
$ conda env list
$ conda activate profile_name #here seed_free_gae_py_3_10
```

## Install libraries
```bash
$ conda install tqdm numpy pytorch torchvision scipy scikit-learn
$ conda install sklearn
$ conda install -c huggingface -c conda-forge datasets
```


## Run the programs
```bash
$ python3 membership_inference_attack.py > ./logs/membership_inference_attack.log 2>&1
$ python3 draw_inference.py > ./logs/draw_inference.log 2>&1
```
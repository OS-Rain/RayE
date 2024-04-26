## Introduction
Thank you for your careful reviewing. To help reviewers better understand our RayE-Sub, we provide source code and partial datasets. You can run our code by following the installation tips below.


## Installation
Our code runs on `Python 3.9` with `PyTorch 1.10.0`, `PyG 2.0.3` and `CUDA 11.3`. Please follow the following steps to create a virtual environment and install the required packages.

1. Create a conda environment:
```
conda create --name RayE python=3.9
conda activate RayE
```

2. Install dependencies:
```
conda install -y pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -r requirements.txt
```

## Reproduce Results
The results of RayE-Sub can be reproduced by running `run_RayE.py`:
```
cd ./src
python run_RayE.py --dataset [dataset_name] --backbone [model_name] --cuda [GPU_id]
```
For example: python run_RayE.py --dataset ba_2motifs  --backbone GIN  --cuda 0

`dataset_name` can be choosen from `ba_2motifs`, `mutag`, `ogbg_molhiv`, `ogbg_molbbbp`, `ogbg_molsider`.

`model_name` can be choosen from `GIN`, `PNA`.

`GPU_id` is the id of the GPU to use. To use CPU, please set it to `-1`.


## Training Logs
Detailed training logs can be found on tensorboard:
```
tensorboard --logdir=./data/[dataset_name]/logs
```

## Hyperparameter Settings
All settings can be found in `./src/configs`.


## Acquiring Datasets
- Ba_2Motifs
    - Raw data files can be downloaded automatically, provided by [PGExplainer](https://arxiv.org/abs/2011.04573) and [DIG](https://github.com/divelab/DIG).

- OGBG-Mol
    - Raw data files can be downloaded automatically, provided by [OGBG](https://ogb.stanford.edu/).

- Mutag
    - Raw data files need to be downloaded manually [here](https://github.com/flyingdoog/PGExplainer/tree/master/dataset), provided by [PGExplainer](https://arxiv.org/abs/2011.04573).
    - Unzip `Mutagenicity.zip` and `Mutagenicity.pkl.zip`.
    - Put the raw data files in `./data/mutag/raw`.

## Notes
Because RayE-Sub needs to precalculate the resistance distance, it may take a long time to process data when you first run it.# RayE

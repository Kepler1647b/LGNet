# LGNet

The source code of article 'Deep learning-based intraoperative differentiation of primary CNS lymphoma and glioma: a discovery, multicenter validation, and proof-of concept study'.

## System requirements
This code was developed and tested in the following settings. 
### OS
- Ubuntu 20.04
### GPU
- Nvidia GeForce RTX 2080 Ti
### Dependencies
- Python (3.9.6)
- Pytorch install = 1.9
- torchvision (0.6)
- CUDA (10.1)
- openslide_python (1.1.1)
- tensorboardX (2.4)
Other dependencies: Openslide (3.4.1), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), pandas (1.0.3), pillow (7.0.0), scikit-learn (0.22.1), scipy (1.3.1)
## Installation guide

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) on your machine (download the distribution that comes with python3).  
  
- After setting up Miniconda, install OpenSlide (3.4.1):  
```
apt-get install openslide-python
```
- Create a conda environment with lgnet.yaml:
```
conda env create -f lgnet.yaml
```  
- Activate the environment:
```
conda activate env1
```
- Typical installation time: 1 hour

## Preprocessing
We use the python files to convert the WSI to patches with size 515*512 pixels and taking color normalization for comparison.
### Slide directory structure
```
DATA_ROOT_DIR/
    └──glioma/
        ├── slide_id.svs
        └── ...
    └──lymphoma/
        ├── slide_id.svs
        └── ...
    ...
```
### Generating patches
- /preprocessing/generate_patch.py
### Color normalization methods
- /preprocessing/StainTools-master/main_staintools.py
- /preprocessing/Vahadane/main.py

## Training and evaluation of LGNet

### Training
```
python train.py --TrainFolder './trainfolder' --NumEpoch 100 --Model resnet50 --Loss 'cross' --LearningRate '0.005' --BatchSize 128 --WeightDecay '0.0005' --Seed 0 --DeviceId '0,1' --FoldN 1
```

The trained checkpoints are saved in /ckpt, 5 files derived from 5-fold cross validation.

### Evaluation
Variate files for evaluation of single fold, ensemble 5 folds for multicenter datasets and proof-of concept dataset are available. Here is an instance for ensembled multicenter dataset evaluation.
```
python test_multicenter_ensemble.py --ModelPath './Model' --DataPath './multicenter1' --ResultPath './result' --DeviceId '0,1' --Model resnet50
```
## Fusion Strategy
LGNet prediction results and pathologist confidence data used in the article are shown in /fusion_data.
Notebook is used to design fusion strategy. 2 files are for proof-of concept dataset and multicenter datasets, respectively.

- fusion_poc.ipynb
- fusion_multicenter.ipynb


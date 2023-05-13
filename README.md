# LGNet

the source code of article 'Deep learning-based intraoperative differentiation of primary CNS lymphoma and glioma: a discovery, multicenter validation, and proof-of concept study'.

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
- Create a conda environment with environment.yaml:
```
conda env create -f environment.yaml
```  
- Activate the environment:
```
conda activate env1
```
- Typical installation time: 1 hour

## Preprocessing
we use the python files to convert the WSI to patches with size 515*512 pixels and taking color normalization for comparison.


- generate_correct2.py
- StainTools-master/main_staintools.py
- Vahadane/main.py

## Training and evaluation of LGNet
files for training and evaluation are in /classification.

for instance:

```
python /classification/train.py --TrainFolder '' --NumEpoch 100 --Model resnet50 --Loss 'cross' --LearningRate '0.005' --BatchSize 128 --WeightDecay '0.0005' --Seed 0 --DeviceId '0,1' --FoldN 1

```

the checkpoints are saved in /ckpt, 5 files derived from 5 fold cross validation.

## Fusion Strategy
LGNet prediction results and pathologist confidence data used in the article are shown in /fusion_data.
notebook is used to design fusion strategy.

- fusion_foresight.ipynb
- fusion_roc_figure.ipynb

for proof-of concept dataset and multicenter datasets, respectively.

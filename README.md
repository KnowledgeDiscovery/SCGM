# Superclass-Conditional Gaussian Mixture Model For Learning Fine-Grained Embeddings (ICLR 2022 Spotlight)

This is the code for the paper "Superclass-Conditional Gaussian Mixture Model for Learning Fine-Grained Embeddings" in ICLR 2022 ([pdf](https://nijingchao.github.io/paper/iclr22_scgm.pdf)). This code provides a demo on BREEDS dataset, and it can be adapted to other datasets including CIFAR-100 and tieredImageNet.

## Requirements
The experiments were done using python3.7, with the following packages:
* learn2learn==0.1.5
* matplotlib==3.4.2
* networkx==2.5.1
* numpy==1.20.3
* pandas==1.3.0
* robustness==1.2.1.post2
* scikit-learn==0.24.2
* scipy==1.7.0
* seaborn==0.11.1
* torch==1.4.0+cu92
* torchvision==0.5.0+cu92

## Datasets
### BREEDS Dataset
1. Download the [ImageNet dataset](https://www.image-net.org/challenges/LSVRC/).
2. Following the [official BREEDS repo](https://github.com/MadryLab/BREEDS-Benchmarks/blob/master/Constructing%20BREEDS%20datasets.ipynb), run
```python
import os
from robustness.tools.breeds_helpers import setup_breeds
info_dir= "[your_imagenet_path]/ILSVRC/BREEDS"
if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
    print("Downloading class hierarchy information into `info_dir`")
    setup_breeds(info_dir)
```
3. The directory structure is
```
└── ILSVRC
    ├── Annotations
    │   └── CLS-LOC
    ├── BREEDS
    │   ├── class_hierarchy.txt
    │   ├── dataset_class_info.json
    │   └── node_names.txt
    ├── Data
    │   └── CLS-LOC
    ├── ImageSets
    │   └── CLS-LOC
    └── Meta
        ├── imagenet_class_index.json
        ├── test.json
        ├── wordnet.is_a.txt
        └── words.txt
```

### CIFAR-100 Dataset
CIFAR-100 can be downloaded from [[link](https://www.cs.toronto.edu/~kriz/cifar.html)].
* Once downloaded, use ``dataset_cifar.py`` in ``dataset/`` folder to generate minibatches for model training.

### TieredImageNet Dataset
TieredImageNet can be downloaded from [[link](https://github.com/renmengye/few-shot-ssl-public)].
* Once downloaded, use ``dataset_tiered_imagenet.py`` in ``dataset/`` folder to generate minibatches for model training.

## Training
First, create a directory to save the pre-trained models.
```
mkdir pretrain_model
```

To train SCGM with a generic encoder (i.e., SCGM-G) on Living17 dataset, run
```
python train_scgm_g.py \
  --data [path to data directory] \
  --workers 32 \
  --epochs 200 \
  --batch_size 256 \
  --hiddim 128 \
  --tau 0.1 \
  --alpha 0.5 \
  --lmd 25 \
  --n-subclass 100 \
  --n-class 17 \
  --dataset living17
```

To train SCGM with a momentum-based encoder (i.e., SCGM-A) on Living17 dataset, run
```
python train_scgm_g.py \
  --data [path to data directory] \
  --arch resnet50 \
  --workers 32 \
  --epochs 200 \
  --batch_size 256 \
  --hiddim 128 \
  --queue multi \
  --metric angular \
  --head-type seq_em \
  --cst-t 0.2 \
  --tau1 0.1 \
  --alpha 0.5 \
  --lmd 25 \
  --n-subclass 100 \
  --n-class 17 \
  --dataset living17
```

The default parameters were set for training on ``BREEDS`` dataset. To check the model parameters, run
```
python train_scgm_g.py -h
```
```
python train_scgm_a.py -h
```

## Testing
To test the performance of the pre-trained SCGM-G on the cross-granularity few-shot (CGFS) learning setting, run
```
python test_scgm_g.py
  --data [path to data directory] \
  --batch_size 256 \
  --n-test-runs 1000 \
  --n-ways 5 \
  --n-shots 1 \
  --n-queries 15 \
  --feat-norm \
  --classifier LR \
  --hiddim 128 \
  --n-subclass 100 \
  --n-class 17 \
  --dataset living17
```

To test the performance of the pre-trained SCGM-A, run
```
python test_scgm_a.py
  --data [path to data directory] \
  --arch resnet50 \
  --batch_size 256 \
  --n-test-runs 1000 \
  --n-ways 5 \
  --n-shots 1 \
  --n-queries 15 \
  --feat-norm \
  --classifier LR \
  --hiddim 128 \
  --n-subclass 100 \
  --n-class 17 \
  --dataset living17
```

Similarly, to test the performance on the fine-grained intra-class setting, run
```
python test_fg_scgm_g.py
```
```
python test_fg_scgm_a.py
```

## Visualization
To visualize the embeddings, include the lines 318 to 340 in ``train_scgm_g.py`` and the lines 341 to 363 in ``train_scgm_a.py``.

## Citation
```
@inproceedings{ni2022superclass,
  title={Superclass-Conditional Gaussian Mixture Model For Learning Fine-Grained Embeddings},
  author={Ni, Jingchao and Cheng, Wei and Chen, Zhengzhang and Asakura, Takayoshi and Soma, Tomoya and Kato, Sho and Chen, Haifeng},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

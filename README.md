# NTFields

>**NTFields: Neural Time Fields for Physics-Informed Robot Motion Planning**
\
>[Ruiqi Ni](https://ruiqini.github.io/),
[Ahmed H Qureshi](https://qureshiahmed.github.io/)


<img src="fig/fig.png" width="663" height="282">

_[Paper](https://openreview.net/forum?id=ApF0dmi1_9K) |
[GitHub](https://github.com/ruiqini/NTFields) |
[arXiv](https://arxiv.org/abs/2210.00120) |
Published in ICLR 2023._

## Introduction

This repository is the official implementation of "NTFields: Neural Time Fields for Physics-Informed Robot Motion Planning". 

## Installation

Clone the repository into your local machine:

```
git clone https://github.com/ruiqini/NTFields --recursive
```

Install requirements:

```setup
conda env create -f NTFields_env.yml
conda activate NTFields
```

Download datasets and pretrained models, exact and put `datasets/` `Experiments/` to the repository directory:

[Datasets and pretrained model](https://drive.google.com/file/d/140W0iOJOwA-nku831mQgPIGGQmXAKtrz/view?usp=share_link)

>The repository directory should look like this:
```
NTFields/
├── datasets/
│   ├── arm/    # 4DOF and 6DOF robot arm, table environment
│   ├── c3d/    # C3D environment
│   ├── gibson/ # Gibson environment
│   └── test/   # box and bunny environment
├── Experiments
│   ├── 4DOF/   # pretrained model for 4DOF arm
│   └── Gib/    # pretrained model for Gibson
•   •   •
•   •   •
```

## Pre-processing

To prepare the Gibson data, run:

```
python dataprocessing/preprocess.py --config configs/gibson.txt
```

To prepare the arm data, run:

```
python dataprocessing/preprocess.py --config configs/arm.txt
```

## Testing

To visualize our path in a Gibson environment, run:

```eval
python test/gib_plan.py 
```

To visualize our path in the 4DOF arm environment, run:

```eval
python test/arm_plan.py 
```

To sample random starts and goals in Gibson environments, run:

```eval
python test/sample_sg.py 
```

To show our statistics result in Gibson environments, run:

```eval
python test/gib_stat.py 
```

## Training

To train our model in a Gibson environment, run:

```train
python train/train_gib.py
```

To train our model in the 4DOF arm environment, run:

```train
python train/train_arm.py 
```

## Citation

Please cite our paper if you find it useful in your research:

```
@inproceedings{
    ni2023ntfields,
    title={{NTF}ields: Neural Time Fields for Physics-Informed Robot Motion Planning},
    author={Ruiqi Ni and Ahmed H Qureshi},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=ApF0dmi1_9K}
}
```

## Acknowledgement
This implementation takes [EikoNet](https://github.com/Ulvetanna/EikoNet) and [NDF](https://github.com/jchibane/ndf/) as references. We thank the authors for their excellent work.


## License

NTFields is released under the MIT License. See the LICENSE file for more details.



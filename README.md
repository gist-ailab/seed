
## SEED
* This repository contains the source codes for the paper "SEED: Self-Error Estimation and Dual Refinement Plug-in for High-Quality Instance Segmentation".
* Implementation of SEED based on [Detectron2](https://github.com/facebookresearch/detectron2)


## Installation
Install Detectron2 first, then install this repo & required packages like below.

```
pip install -e .
pip install --upgrade "protobuf<=3.20.1" 
pip install torch_dct
pip install monai==1.0.0
pip install opencv-python 
pip install setuptools==59.5.0
pip install fvcore

```


## RUN
Setting datasets with detectron2 format and then run bash files below.
```
sh bash/train.sh
sh bash/eval.sh
```


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

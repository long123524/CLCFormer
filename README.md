# CLCFormer

Official Pytorch Code base for [Integrating spatial details with long-range contexts for semantic segmentation of very high resolution remote sensing images]

[Project](https://github.com/long123524/CLCFormer)

## Introduction

This paper presents a cross-learning network (i.e., CLCFormer) integrating fine-grained spatial details within long-range global contexts based upon convolutional neural network (CNN) and transformer, for semantic segmentation of very high-resolution (VHR) remote sensing images. 

## Using the code:

The code is stable while using Python 3.7.0, CUDA >=11.0

- Clone this repository:
```bash
git clone https://github.com/long123524/CLCFormer
cd CLCFormer
```

To install all the dependencies using conda or pip:

```
PyTorch
timm
OpenCV
numpy
tqdm
PIL
```

## Datasets

Inria building dataset:https://project.inria.fr/aerialimagelabeling/
WHU building dataset:http://gpcv.whu.edu.cn/data/building_dataset.html
Potsdam dataset:https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx


## Training and testing
Will be coming soon!

All codes will be release after our paper accepted.

## Acknowledgement
We are very grateful for these excellent works [ST-UNet](https://github.com/XinnHe/ST-UNet), [TransFuse](https://github.com/Rayicer/TransFuse) and [BuildFormer](https://github.com/WangLibo1995/BuildFormer), which have provided the basis for our framework.

### Citation:
```
Long J, Li M, Wang X. Integrating spatial details with long-range contexts for semantic segmentation of very high resolution remote sensing images[J]. IEEE GEOSCIENCE AND REMOTE SENSING LETTERS.
```

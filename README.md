# Improving DINOv2’s spatial representations in 19 GPU hours with Patch Neighbor Consistency

**University of Amsterdam**

Valentinos Pariza*, 
Mohammadreza Salehi*, 
Gertjan Burghouts, 
Francesco Locatello,
Yuki M. Asano 

 [[`Paper`](https://arxiv.org/abs/2408.11054)] 


## Table of Contents

- [Introduction](#introduction)
- [GPU Requirements](#gpu-requirements)
- [Training](#training-setup)
- [Evaluation](#evaluation)
- [Dataset Preparation](#datasets)
- [Visualizations](#visualizations)
- [Citation](#citation)



## Introduction

NeCo introduces a new self-supervised learning technique for enhancing spatial representations in vision transformers. By leveraging Patch Neighbor Consistency, NeCo captures fine-grained details and structural information that are crucial for various downstream tasks, such as semantic segmentation.

<p align="center">
  <img src="Images/Neco.jpg" alt="NeCo Overview" width="800"/>
</p>



Key features of NeCo include:
1. Patch-based neighborhood consistency
2. Improved dense prediction capabilities
3. Efficient training requiring only 19 GPU hours
4. Compatibility with existing vision transformer backbone


Below is a table with some of our results on Pascal VOC 2012 based on DINOv2 backbone. 
<table>
  <tr>
    <th>backbone</th>
    <th>arch</th>
    <th>params</th>
    <th>Overclustering k=500</th>
    <th>Dense NN Retrieval</th>
    <th>linear</th>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>DINOv2</td>
    <td>ViT-S/14</td>
    <td>21M</td>
    <td>72.6</td>
    <td>81.3</td>
    <td>78.9</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EWvXdau9r6NIr-vIc_xDlxAB1sDrljoaPR_A3JhIEeE8dw?e=pOXEXG">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EYUuXDWfsd1Pi_MgkYcEVfcB_wVWeIr9faBO5xdc4L4REA?e=BW5aLc">teacher</a></td>
  </tr>
  <tr>
    <td>DINOv2</td>
    <td>ViT-B/14</td>
    <td>85M</td>
    <td>71.8</td>
    <td>83.3</td>
    <td>81.4</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EQGCG4KgPW9Eqgl_xcwFcgsBitTXIOL1GfGcUa1MJq4cUw?e=WQb7R7">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EdR_Fily3xNInI-QlW2MGIUBt3DdDoMD8dGvvDDFn8wiPQ?e=oFaZhp">teacher</a></td>
  </tr>
  <tr>
    <td>DINO</td>
    <td>ViT-S/16</td>
    <td>22M</td>
    <td>47.9</td>
    <td>61.3</td>
    <td>65.8</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EX1_VUd7oxRErMXqOUcjSGcBqut9vGER8aqtGz9Yvl4_pQ?e=9VMSfb">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EfTNadBzg0VNrjmAoWDe2rsBVjPVcGvnM-rKkFAYq14IgQ?e=45j3L1">teacher</a></td>
  </tr>
  <tr>
    <td>TimeT</td>
    <td>ViT-S/16</td>
    <td>22M</td>
    <td>53.1</td>
    <td>66.5</td>
    <td>68.5</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EUW5Ym_h2rlMiuU8UBRdi2IBOCicWyFbUpQL46FSjqzJdg?e=2TmaQB">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EZ4bSpcUEYtHtsdusEiR08cBikV63UzmrCuIgLNPeRqEew?e=U4Y2ga">teacher</a></td>
  </tr>
  <tr>
    <td>Leopart</td>
    <td>ViT-S/16</td>
    <td>22M</td>
    <td>55.3</td>
    <td>66.2</td>
    <td>68.3</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EZ3DX9ftK41Gik6ZLz-iPjgB2dZcW5Xco4B83_wM2dTy2A?e=885yjU">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/Ed8Bp7b3I5ZCs6ZPIC2vUGEBpTOPG_Avd0nL6eW6hVZWdQ?e=UZvpor">teacher</a></td>
  </tr>
</table>

In the following sections, we will delve into the training process, evaluation metrics, and provide instructions for using NeCo in your own projects.

## GPU Requirements

<a name="gpu-requirements"></a>
Optimizing with our model, ***NeCo***, does not necessitate a significant GPU budget. Our training process is conducted on a single NVIDIA A100 GPU.


## Environment Setup
We use conda for dependency management. 
Please use `environment.yml` to install the environment necessary to run everything from our work. 
You can install it by running the following command:
```bash
conda env create -f environment.yaml
```

#### Pythonpath
Export the module to PYTHONPATH within the repository's parent directory.
`
export PYTHONPATH="${PYTHONPATH}:PATH_TO_REPO"
`
#### Neptune
We use neptune for logging experiments.
Get you API token for neptune and insert it in the corresponding run-files.
Also make sure to adapt the project name when setting up the logger.



## Loading pretrained models

To use NeCo embeddings on downstream dense prediction tasks, you just need to install `timm`  and `torch` and run:

```python
import torch
path_to_checkpoint = "<your path to downloaded ckpt>"
model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
state_dict = torch.load(path_to_checkpoint)
model.load_state_dict(state_dict, strict=False)
```

## Training Setup

### Repository Structure

- `src/`: Model, method, and transform definitions
- `experiments/`: Scripts for setting up and running experiments
- `data/`: Data modules for ImageNet, COCO, Pascal VOC, and ADE20k

### Training with NeCo

- Use configs in `experiments/configs/` to reproduce our experiments
- Modify paths in config files to match your dataset and checkpoint directories
- For new datasets:
  1. Change the data path in the config
  2. Add a new data module
  3. Initialize the new data module in `experiments/train_with_neco.py`

For instance, to start a training on COCO:

```bash
python experiments/train_with_neco.py --config_path experiments/configs/neco_224x224.yml
```

## Evaluation

We provide several evaluation scripts for different tasks. For detailed instructions and examples, please refer to the [Evaluation README](evaluation_README.md). Here's a summary of the evaluation methods:

1. **Linear Segmentation**: 
   - Use `linear_finetune.py` for fine-tuning.
   - Use `eval_linear.py` for evaluating on the validation dataset.

2. **Overclustering**:
   - Use `eval_overcluster.py` to evaluate overclustering performance.

3. **Cluster Based Foreground Extraction + Community Detection (CBFE+CD)**:
   - Requires downloading noisy attention train and val masks.
   - Provides examples for both ViT-Small and ViT-Base models.

Each evaluation method has specific configuration files and command-line arguments. The Evaluation README provides detailed examples and instructions for running these evaluations on different datasets and model architectures.


## Datasets

We use PyTorch Lightning data modules for our datasets. Supported datasets include ImageNet100k, COCO, Pascal VOC, and ADE20k. Each dataset requires a specific folder structure for proper functioning.

Data modules are located in the `data/` directory and handle loading, preprocessing, and augmentation. When using these datasets, ensure you update the paths in your configuration files to match your local setup.

For detailed information on dataset preparation, download instructions, and specific folder structures, please refer to the [Dataset README](dataset_README.md).



##  Visualizations

We provide visualizations to help understand the performance of our method. Below is an example of Cluster-Based Foreground Extraction (CBFE) results on the Pascal VOC dataset:

![CBFE Visualization](Images/cbfe.png)

This visualization shows the ability of NeCo without relying on any supervision. Different objects are represented by distinct colors, and the method captures tight and precise object boundaries.

## Citations

<a name="citation"> </a>

If you find this repository useful, please consider giving a star ⭐ and citation 📣:
``` 
@inproceedings{
   pariza2025near,
   title={Near, far: Patch-ordering enhances vision foundation models' scene understanding},
  author={Valentinos Pariza and Mohammadreza Salehi and Gertjan J. Burghouts and Francesco Locatello and Yuki M Asano},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?     id=Qro97zWC29}
}

```

#### Note: Our repository is developed by adopting and adapting multiple parts of the [Leopart](https://github.com/MkuuWaUjinga/leopart) model, as well as parts from other works like DINOv2, DINO, R-CNN, ...






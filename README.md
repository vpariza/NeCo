# Improving DINOv2’s spatial representations in 16 GPU hours with Patch Neighbor Consistency

#### Note: Our repository is developed by adopting and adapting multiple parts of the [Leopart](https://github.com/MkuuWaUjinga/leopart) model, as well as parts from other works like DINOv2, DINO, R-CNN, ...
### Overview of NeCo Models on DINOv2

Below is a table with some of our results on Pascal VOC 2012. 
<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>Overclustering k=500</th>
    <th>Dense NN Retrieval</th>
    <th>linear</th>
    <th colspan="2">download</th>
    
  </tr>
  <tr>
    <td>ViT-S/14</td>
    <td>21M</td>
    <td>72.6</td>
    <td>81.3</td>
    <td>78.9</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EWvXdau9r6NIr-vIc_xDlxAB1sDrljoaPR_A3JhIEeE8dw?e=pOXEXG">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EYUuXDWfsd1Pi_MgkYcEVfcB_wVWeIr9faBO5xdc4L4REA?e=BW5aLc">teacher</a></td>
  </tr>
  <tr>
    <td>ViT-B/14</td>
    <td>85M</td>
    <td>71.8</td>
    <td>83.3</td>
    <td>81.4</td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EQGCG4KgPW9Eqgl_xcwFcgsBitTXIOL1GfGcUa1MJq4cUw?e=WQb7R7">student</a></td>
    <td><a href="https://1drv.ms/u/c/67fac29a77adbae6/EdR_Fily3xNInI-QlW2MGIUBt3DdDoMD8dGvvDDFn8wiPQ?e=oFaZhp">teacher</a></td>
  </tr>
</table>



To use NeCo embeddings on downstream dense prediction tasks, you just need to install `timm`  and `torch` and run:

```python
import torch
path_to_checkpoint = "<your path to downloaded ckpt>"
model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
state_dict = torch.load(path_to_checkpoint)
model.load_state_dict(state_dict, strict=False)
```

### Setup

#### Repo Structure
`src/`: Contains model, method and transform definitions 

`experiments/`: Contains all scripts to setup and run experiments. 

`data/`: Contains data modules for ImageNet, COCO, Pascal VOC and Ade20k

#### Environment
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

#### Fine-Tune with NeCo
Note that you can change the config according to your needs. For fine-tuning on a new dataset, you would have to 
change the data path, add a data module and initialize it in the [train_with_neco.py](./experiments/train_with_neco.py).

If you want to reproduce our experiments you can use the configs in `experiment/configs/`. 
You just need to adapt the paths to your datasets and checkpoint directories accordingly.
Below you can see an example of finetuning DINOv2 with our method to get NeCo:

Train on COCO could look like this:
```bash
python experiments/train_with_neco.py --config_path experiments/configs/neco_224x224.yml
```

#### Evaluation

##### Evaluation: Linear Segmentation
For linear segmentation fine-tuning we provide a script as well. All configs can be found in [`experiments/linear_segmentation/configs/`](experiments/linear_segmentation/configs/). 
An exemplary call to evaluate NeCo trained on Pascal VOC 2012 could look like this:
```
python experiments/linear_segmentation/linear_finetune.py --config_path experiments/linear_segmentation/configs/pascal/neco.yml
```
We also provide a script to evaluate on the validation dataset at [`experiments/linear_segmentation/eval_linear.py`](experiments/linear_segmentation/eval_linear.py).

##### Evaluation: Overclustering
For overclustering evaluation we provide a script `eval_overcluster.py` under `experiments/overcluster`.
An exemplary call to evaluate a NeCo trained on Pascal VOC 2012 on Pascal VOC could look like this:
```
python experiments/overcluster/eval_overcluster.py --config_path experiments/overcluster/configs/pascal/neco.yml
```

##### Evaluation: Clster Based Foreground Extraction + Community Detection
Please download the noisy attention train and val masks into ` --save_folder` before running the script. By default the ` --save_folder`  folder is the `./embeddings`, so you can create one and add the files there, or pass the path to the folder you like to use as argument to the parameter `--save_folder` in the execution below. 
Therefore for ViT, you can execute the following command to reproduce our CBFE+CD results. Furthermore, you need to provide yout path to the Pascal VOC root folder by providing an argument for `--data_dir`.
Firstly, you can try with the ViT-Small version of our model trained from DINOv2:
```bash
python experiments/fully_unsup_seg/fully_unsup_seg.py --ckpt_path <PATH TO VIT_SMALL CHECKPOINT>  --experiment_name vit14s --b
est_k 189 --best_mt 1.2 --best_wt 0.07 --arch vit-small --data_dir <YOUR PATH>

```
And you can also try with a ViT-Base size as well trained from DINOv2:
```bash
python experiments/fully_unsup_seg/fully_unsup_seg.py --ckpt_path <PATH TO VIT_BASE CHECKPOINT>  --experiment_name vit14s --b
est_k 149 --best_mt 1.7 --best_wt 0.07 --arch vit-base --save_folder ./embeddings_base --data_dir <YOUR PATH>
```


#### Dataset Setup
The data is encapsulated in lightning data modules. 
Please download the data and organize them in the folders are indicated below.
You can follow the section from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md)
to download and setup the datasets. We expect the following dataset folder structure.

##### Imagnet100k
The structure should be as follows:
```
dataset root.
│   imagenet100.txt
└───train
│   └─── n*
│       │   *.JPEG
│       │   ...
│   └─── n*
│    ...
```
The path to the dataset root is to be used in the configs.

##### COCO
The structure for training should be as follows:
```
dataset root.
└───images
│   └─── train2017
│       │   *.jpg
│       │   ...
```

The structure for evaluating on COCO-Stuff and COCO-Things should be as follows:
```
dataset root.
└───annotations
│   └─── annotations
│       └─── stuff_annotations
│           │   stuff_val2017.json
│           └─── stuff_train2017_pixelmaps
│               │   *.png
│               │   ...
│           └─── stuff_val2017_pixelmaps
│               │   *.png
│               │   ...
│   └─── panoptic_annotations
│       │   panoptic_val2017.json
│       └─── semantic_segmenations_train2017
│           │   *.png
│           │   ...
│       └─── semantic_segmenations_val2017
│           │   *.png
│           │   ...
└───coco
│   └─── images
│       └─── train2017
│           │   *.jpg
│           │   ...
│       └─── val2017
│           │   *.jpg
│           │   ...
```
##### VOC Pascal
The structure for training and evaluation should be as follows:
```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───images
│   │   *.jpg
│   │   ...
└───sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt
```

##### ADE20k
The structure for training and evaluation should be as follows:
```
dataset root.
├── ADEChallengeData2016
│   ├── annotations
│   │   ├── training
│   │   ├── validation
│   ├── images
│   │   ├── training
│   │   ├── validation
```



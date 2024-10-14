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
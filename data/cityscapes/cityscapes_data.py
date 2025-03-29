import os
import torch
import numpy as np

from PIL import Image
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

class Cityscapes(Dataset):

    def __init__(self, root, transforms, split='train', file_set=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split
        
        assert os.path.exists(self.root), "Please setup the dataset properly"
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.inv_index = 255
        inv = self.inv_index
        self._key = np.array([inv, inv, inv, inv, inv, inv,
                              inv, inv, 0, 1, inv, inv,
                              2, 3, 4, inv, inv, inv,
                              5, inv, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              inv, inv, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            # check that all classes are in the expected dataset classes
            assert (value in self._mapping) 
        # map every class to an index such that all the valid classes are in [0, 18] and invalid ones are -1
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        # Load
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.mask_paths[index])

        target = self._mask_transform(target).float() / 255.0
        target = F.to_pil_image(target)

        # Augment
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            # Convert unwanted class to the class to skip
            # which in our case we always skip the class of 255
        
        return image, target

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths

class CityscapesDataModule(pl.LightningDataModule):

    def __init__(self,
                 root,
                 train_transforms,
                 val_transforms,
                 shuffle,
                 num_workers,
                 batch_size,
                 val_file_set=None,
                 train_file_set=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.val_file_set = val_file_set
        self.train_file_set = train_file_set

    def setup(self, stage: Optional[str] = None):
        self.val = Cityscapes(self.root, self.val_transforms, split='val', file_set=self.val_file_set)
        self.train = Cityscapes(self.root, self.train_transforms, split='train', file_set=self.train_file_set)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.train)

    def get_val_dataset_size(self):
        return len(self.val)
    
    def get_num_classes(self):
        return 19

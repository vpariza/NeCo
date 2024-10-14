import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

from data.imagenet import logger

class TrainDataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset: Dataset,
                 num_workers: int,
                 batch_size: int):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if stage == 'fit' or stage is None:
            print(f"Train size {len(self.dataset)}")
        else:
            raise NotImplementedError("There is no dedicated val/test set.")
        logger.info(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)
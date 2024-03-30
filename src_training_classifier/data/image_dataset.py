from torchvision.datasets import ImageFolder
from os.path import join

from torchvision import transforms as T
import torch
from typing import Tuple, Union, Optional, Callable
from torch import randperm, Generator
from copy import deepcopy
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform(mean: Tuple = IMAGENET_MEAN,
                        std: Tuple  = IMAGENET_STD,
                        resize_size: Union[int, Tuple[int, int]] = 256, 
                        crop_size:   Union[int, Tuple[int, int]] = 224,
                        pad: Tuple[int, int, int, int] = [0, 0, 0, 0]) \
                            -> Callable:
    """ It returns a standard basic transform for training. """

    return  T.Compose([T.Pad(pad),
                       T.Resize(resize_size),
                       T.RandomCrop(crop_size),
                       T.RandomHorizontalFlip(),
                       T.ToTensor(),
                       T.Normalize(mean=mean, std=std)])


def get_test_transform(mean: Tuple = IMAGENET_MEAN,
                       std: Tuple  = IMAGENET_STD,
                       resize_size: Union[int, Tuple[int, int]] = 256, 
                       crop_size:   Union[int, Tuple[int, int]] = 224,
                       pad: Tuple[int, int, int, int] = [0, 0, 0, 0]) \
                            -> Callable:
    """ It returns a standard basic transform for test. """

    return  T.Compose([T.Pad(pad),
                       T.Resize(resize_size),
                       T.CenterCrop(crop_size),
                       T.ToTensor(),
                       T.Normalize(mean=mean, std=std)])


class ImageFolderIdx(ImageFolder):
    """ Wrapper of ImageFolder that returns, additionally, the indices of images. """

    def __init__(self, 
                 root: str, 
                 transform: Optional[Callable] = None,
                 *,
                 image_key: str = "images",
                 label_key: str = "labels",
                 idx_key: str = "idxs",
                 csv_labels: Optional[str] = None):
        """
            Args:
                root (str): path to the dataset.
                transform (Callable, optional): the transform to be used.
                image_key (str): the key in the dictionary returned by the dataset for
                the images.
                label_key (str): the key in the dictionary returned by the dataset for
                the labels.
                idx_key (str): the key in the dictionary returned by the dataset for
                the idxs.
                custom_label_key (str): the key in the dictionary returned by the dataset 
                for custom labels.
                csv_labels (str, optional): path to an optional csv file containing in 
                each row: image_name.fmt, #class

        """
        super().__init__(root, transform)

        # keys in the dict returned by the dataset
        self.image_key = image_key
        self.label_key = label_key
        self.idx_key = idx_key

        # if a csv files is given, the labels are loaded.
        self._load_labels_from_csv(csv_path=csv_labels)

        self.indices = torch.tensor([i for i in range(super().__len__())])


    def _load_labels_from_csv(self, csv_path: str):
        """ Loads labels from a csv file. """

        if csv_path is None: return

        # read the csv and split each row using , as separator (each line is in the 
        # format relative_path, class)
        with open(csv_path, "r") as f:
            lines = [l.split(",") for l in f.readlines()]
        
        # update samples and images with new labels
        self.samples = [(join(self.root, name.strip()), int(class_n.strip())) \
                        for name, class_n in lines]
        self.imgs = self.samples

        # update targets with new labels
        self.targets = [s[1] for s in self.samples]

        self.classes = np.unique(self.targets)


    def split(self, train_per: float, seed: int) -> Tuple:
        """
            Splits the dataset into train/val

            Args:
                train_per (float): the percentage (between 0 and 1) of samples to be used
                in the training set. (1-train)*100% samples will be used in the
                validation set.
                seed (int): the seed to use for splitting the dataset.
        """
        length = super().__len__()
        train_length =  int(length * train_per)

        indices = randperm(length, generator=Generator().manual_seed(seed)).tolist() 

        train_dataset = deepcopy(self)
        test_dataset  = deepcopy(self)

        train_dataset.indices = indices[0: train_length]
        test_dataset.indices  = indices[train_length:]

        return train_dataset, test_dataset


    def __len__(self) -> int:
        """ Returns the length of the dataset. """
        return len(self.indices)


    def __getitem__(self, index: int) -> dict:
        """ Get the item of the current dataset at a given index. Returns a dict 
        containing the image, the label and the idx of the sample. """
        real_index = self.indices[index]
        image, label = super().__getitem__(real_index)

        return {self.image_key: image, 
                self.label_key: label, 
                self.idx_key: index}


    def set_transform(self, transform: Callable) -> None:
        """ Set the given transform for the current dataset. """
        self.transform = transform

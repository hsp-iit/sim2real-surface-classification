from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader, Dataset

from composed_model import ComposedModel

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch
import numpy as np

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from typing import Union, Optional


@torch.inference_mode()
def eval(dataset: Dataset,
         batch_size: int, 
         model: Union[ComposedModel, DDP],
         accelerator: Accelerator,
         *,
         get_features: bool = False,
         get_backbone_features: bool = False,
         get_logits: bool = False,
         get_labels: bool = False,
         get_predictions: bool = False,
         model_in_eval_mode: bool = True,
         labels_map: Optional[torch.Tensor] = None,
         n_classes=None) -> dict:
    """
        It tests a model (with bottleneck and classifier).
        Additionally features and logits can be returned.

        Args:
            dataset (Dataset): dataloader to use.
            batch_size (int): the global batch size to use.
            model (ComposedModel): the full model (ComposedModel), possibly wrapped 
            by a DDP.
            accelerator (Accelerator): the accelerator.
            get_features (bool): True to return the features extracted (after the 
            bottleneck).
            get_backbone_features (bool): True to return the features extracted 
            (before the bottleneck).
            get_logits (bool): True to return the logits.
            get_labels (bool): True to return the labels.
            get_predictions (bool): True to return the predictions.
            model_in_eval_mode (bool): set model to eval mode before doing the validation.

        Returns:
            A dictionary containing the results informations
    """
    
    output_dict = {}

    # info from accelerator
    device = accelerator.device
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    n_images  = len(dataset) 

    if n_classes is None: 
        n_classes = len(dataset.classes)
   
    if labels_map is not None: 
        n_classes = len(torch.unique(labels_map))
        labels_map = labels_map.to(device)

    #accelerator.print(f"n_classes: {n_classes}")
    #accelerator.print(f"labels_map: {labels_map}")

    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=False,
                        drop_last=False, 
                        num_workers=cpu_count())
    
    loader = prepare_data_loader(dataloader=loader, 
                                 device=device, 
                                 num_processes=num_processes, 
                                 process_index=process_index, 
                                 split_batches=True,
                                 even_batches=False, 
                                 put_on_device=True)

    confusion_matrix = ConfusionMatrix(n_classes=n_classes, accelerator=accelerator)

    # initialize tensor to save features if necessary
    if get_features:
        if isinstance(model, DDP):
            features_dim = model.module.get_features_dim()
        else:
           features_dim = model.get_features_dim()

        output_dict["features"] = torch.zeros([n_images, features_dim], device=device)

    # initialize tensor to save backbone features if necessary
    if get_backbone_features:
        if isinstance(model, DDP):
            backbone_features_dim = model.module.get_backbone_features_dim()
        else:
           backbone_features_dim = model.get_backbone_features_dim()

        output_dict["backbone_features"] = torch.zeros([n_images, backbone_features_dim], 
                                                        device=device)

    # initialize tensor to save logits if necessary
    if get_logits:
        output_dict["logits"] = torch.zeros([n_images, n_classes], device=device)

    if get_labels:
        output_dict["labels"] = torch.zeros([n_images], dtype=torch.long, device=device)

    if get_predictions:
        output_dict["predictions"] = torch.zeros([n_images], dtype=torch.long, device=device)


    # modules in evaluation mode
    if model_in_eval_mode: model = model.eval()

    for batch in loader:
        
        # unpack inputs
        images, idxs, labels = batch["images"], batch["idxs"], batch["labels"]

        outputs = model(images)                      

        # unpack outputs
        logits, features, backbone_features = \
        outputs["logits"], outputs["features"], outputs["backbone_features"]
    
        if get_features:
            output_dict["features"][idxs] = features
        
        if get_backbone_features:
            output_dict["backbone_features"][idxs] = backbone_features

        if get_logits:
            output_dict["logits"][idxs] = logits

        if get_labels:
            output_dict["labels"][idxs] = labels.clone().detach() 

        # hard predictions (argmax) to evaluate accuracy
        _, predictions = torch.max(logits, dim=1)
        
        if get_predictions:
            output_dict["predictions"][idxs] = predictions 
            
        if labels_map is not None: predictions = labels_map[predictions]

        #accelerator.print(f"predictions: {predictions}")

        confusion_matrix.update(predictions, labels)

    # reduce all tensors across gpus

    if accelerator.num_processes > 1:
        confusion_matrix.reduce_across_processes()

        for k in output_dict: 
            output_dict[k] = accelerator.reduce(output_dict[k])
    
    accuracy_micro, accracy_macro, accuracy_per_class = confusion_matrix.compute()

    output_dict["accuracy_micro"] = float(accuracy_micro)
    output_dict["accuracy_macro"] = float(accracy_macro) 
    output_dict["accuracy_per_class"] = accuracy_per_class
    output_dict["confusion_matrix"] = confusion2str(confusion_matrix.matrix)

    model = model.train()

    return output_dict 


class ConfusionMatrix:
    def __init__(self, n_classes: int, accelerator: Accelerator):

        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.n_classes = n_classes
        self.matrix = torch.zeros((n_classes, n_classes), 
                                  dtype=torch.int64, 
                                  device=self.device)

    def update(self, predictions, target):
        # ROWS = TARGET
        # COLS = PREDICTIONS

        n = self.n_classes
        k = (target >= 0) & (target < self.n_classes)
        inds = n * target[k].to(torch.int64) + predictions[k]
        self.matrix += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.matrix.zero_()

    def reduce_across_processes(self):
        if self.accelerator.num_processes > 1:
            self.matrix = self.accelerator.reduce(self.matrix)

    def compute(self):

        matrix = self.matrix.float()
        accuracy_micro = torch.diag(matrix).sum() / matrix.sum()
        accuracy_per_class = torch.diag(matrix) / matrix.sum(1)
        accuracy_macro = torch.mean(accuracy_per_class)

        return accuracy_micro, accuracy_macro, accuracy_per_class
    


def _convert_matrix(matrix, max_length=3):
    output = []

    for row in matrix:
        current_row = []
        for value in row:
            value = str(value)
            n_spaces = max_length - len(value)
            value = value + " " * n_spaces
            current_row.append(value)
        output.append(current_row)
    return output

def _get_first_row(labels, max_length=3):
    output = []
    for l in labels:
        l = str(l)
        n_spaces = max_length - len(l)
        l = l + " " * n_spaces
        output.append(l)

    return "".join(output)

def confusion2str(matrix, labels=None, max_length=5):


    if isinstance(matrix, torch.Tensor): 
        matrix = matrix.cpu()
        matrix = np.array(matrix)

    if labels is None:
        labels = [str(i) for i in range(len(matrix))]

    output = ""

    space = " " * max_length

    first_row = space + " | " + _get_first_row(labels, max_length=max_length)
    second_row = "-" * len(first_row)

    output += first_row  + "\n"
    output += second_row + "\n"

    matrix   = _convert_matrix(matrix, max_length=max_length)

    for row, l in zip(matrix, labels):
        current_space = " " * (max_length - len(l))
        output += current_space + l + " | " + "".join(row) + "\n"
    
    return output
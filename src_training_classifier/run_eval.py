import yaml
from fire import Fire
from os.path import join

from composed_model import ComposedModel
from data import get_test_transform, ImageFolderIdx
from evaluator import eval
from accelerate import Accelerator

import torch


def load_hparms(path: str) -> dict:
    """ Loads the hparms from a yaml file and returns a dict. """

    with open(path, "r") as f: 
        hparms = yaml.full_load(f)

    return hparms

def override(hparams, override_dict) -> dict:
    for k,v in override_dict.items():
        if k in hparams: hparams[k] = v
    return hparams



def main(config: str, ckpt: str, data_path: str, log=None):
    
    accelerator = Accelerator(split_batches=True)

    # 1) load config
    accelerator.print("Loading hparams...")

    hparams = load_hparms(config)
    hparams_model = hparams["model"]
    hparams_data  = hparams["data"]

    # 2) model
    accelerator.print("Loading model...")

    model = ComposedModel(**hparams_model)

    ckpt_state_dict = torch.load(ckpt, map_location="cpu")["model"]
    model.load_state_dict(ckpt_state_dict, strict=True)

    model = accelerator.prepare(model)

    # 3) data
    accelerator.print("Loading data...")

    transform = get_test_transform(**hparams_data)
    dataset = ImageFolderIdx(root=data_path, 
                             transform=transform, 
                             csv_labels=join(data_path, "labels.csv"))
    
    # 4) accelerator
    
    accelerator.print("Starting validation...")
    test_results = eval(dataset=dataset, 
                        model=model, 
                        batch_size=64, 
                        accelerator=accelerator,
                        get_predictions=True,
                        get_labels=True)
    
    accuracy = test_results["accuracy_micro"]
    confusion_matrix = test_results["confusion_matrix"]

    accelerator.print(f"Test accuracy: {accuracy*100:.2f}\n\n")
    accelerator.print(confusion_matrix)
    
    if accelerator.is_main_process and log is not None:

        real_paths  = [l[0] for l in dataset.samples]
        #real_labels = [l[1] for l in dataset.samples]

        with open(log, "a+") as f:
            for path, l1, pred in zip(real_paths, test_results["labels"], test_results["predictions"]):
                print(f"{path},{l1},{pred}", file=f)


if __name__ == "__main__":
    Fire(main)
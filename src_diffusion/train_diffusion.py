from pytorch_diffusion import Unet, Trainer, GaussianDiffusion

from fire import Fire
import yaml
from accelerate import Accelerator
from pathlib import Path
from os.path import join, isfile
from os import listdir
import torchvision.transforms as T

from typing import List


def load_hparms(path: str) -> dict:
    """ Loads the hparms from a yaml file and returns a dict. """

    with open(path, "r") as f: hparms = yaml.safe_load(f)
    
    return hparms


def override_hparams(hparams, accelerator, **override_h):
    """ Overrides hparams with custom ones """

    for k, v in override_h.items():
        if k in hparams: 
            accelerator.print(f"override hparam: {k} = {v}")
            hparams[k] = v
    
    return hparams

def find_last_ckpt(path, base_name="model"):
    """ Find the last available checkpoint. """
    # all candidate files
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [f for f in files if base_name in f]

    # zero candidate ckpt
    if len(files) == 0: return None

    # keep just the numbers of the ckpt
    files = [int(''.join(c for c in f if c.isdigit())) for f in files]
    files = sorted(files)
    
    return files[-1]


def train(dataset_path: str, 
          out_path: str, 
          config_path: str,
          condition_folder: str = None,
          **override_h):
    """
        Train diffusion model from config file. 

        Args:
            dataset_path (str): the dataset path.
            out_path (str): the output path where to save ckpts and samples
            config_path (str): the config file for training.
            condition_folder (str): the path to the condition folder. If None not 
            condition will be used.
            override_h: additional hparams that will override the one in the config.
    """
        
    accelerator  = Accelerator()
    all_hparams  = load_hparms(config_path)

    # get hparams subsections from config
    unet_hparams      = all_hparams["unet"]
    diffusion_hparams = all_hparams["diffusion"]
    trainer_hparams   = all_hparams["trainer"]

    # update relevant hparams
    trainer_hparams["folder"]           = dataset_path
    trainer_hparams["condition_folder"] = condition_folder
    trainer_hparams["results_folder"]   = out_path

    if accelerator.is_main_process:
        Path(out_path).mkdir(parents=True, exist_ok=True)

    # override custom hparams
    unet_hparams      = override_hparams(unet_hparams, accelerator, **override_h)
    diffusion_hparams = override_hparams(diffusion_hparams, accelerator, **override_h)
    trainer_hparams   = override_hparams(trainer_hparams, accelerator, **override_h)


    # initialize unet, diffusion and trainer
    unet      = Unet(**unet_hparams)
    diffusion = GaussianDiffusion(unet, **diffusion_hparams) 
    trainer   = Trainer(diffusion, **trainer_hparams)


    ckpt = find_last_ckpt(path=out_path)

    if ckpt: 
        accelerator.print(f"Found exisiting ckpt: {ckpt}")
        trainer.load(ckpt)

    # start training
    trainer.train()


if __name__ == "__main__":
    Fire(train)
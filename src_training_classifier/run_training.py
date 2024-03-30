import yaml
from fire import Fire
from composed_model import ComposedModel
from trainer import Finetuner


def load_hparms(path: str) -> dict:
    """ Loads the hparms from a yaml file and returns a dict. """

    with open(path, "r") as f: 
        hparms = yaml.full_load(f)

    return hparms

def override(hparams, override_dict) -> dict:
    for k,v in override_dict.items():
        if k in hparams: hparams[k] = v
    return hparams

def train(config: str,
          config_model: str,
          data_src: str,
          data_tgt: str,
          n_classes: int,
          labels_file: str,
          inner_folder: str = None,
          **kwargs):

    print("Starting training!")

   # load model hparams and override the ones passed as kwargs
    hparams_model = load_hparms(config_model)
    hparams_model = override(hparams_model, kwargs)


    # load method hparams and override the ones passed as kwargs
    hparams = load_hparms(config)    
    hparams = override(hparams, kwargs)
    
    # get model name (to save ckpt with correct name)
    model_name = hparams_model["model_name"]
    seed       = hparams["seed"]
    ckpt_name = f"{model_name}_{seed}.ckpt"
    

    hparams["data_src"]        = data_src
    hparams["data_tgt"]        = data_tgt
    hparams["inner_folder"]    = inner_folder
    hparams["labels_file"]     = labels_file

    hparams_model["n_classes"] = n_classes

    # create model and finetuner

    print("Creating model")
    model = ComposedModel(**hparams_model)

    print("Creating Trainer with DANN.")
    finetuner = Finetuner(model=model, 
                          ckpt_name=ckpt_name, 
                          **hparams)
    # run
    print("train")
    finetuner.train()

if __name__ == "__main__":
    print("starting main!")
    Fire(train)
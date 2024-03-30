from pytorch_diffusion import Unet, GaussianDiffusion
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
import torchvision.transforms as T

from PIL import Image

from accelerate import Accelerator
from pathlib import Path
from tqdm import tqdm 
from fire import Fire
import yaml
import os

from typing import List

class ImageFolderIdx(Dataset):
    """ Dataset that returns pairs: IMAGE IDX """

    def __init__(self,
                 root: str,
                 image_size: int,
                 padding: List = [0, 0, 0, 0], # left, top, right and bottom
                 condition_folder: str = None,
                 exts: List = ['jpg', 'jpeg', 'png', 'tiff']):
        
        super().__init__()

        self.root = root
        self.image_size = image_size
        self.condition_folder = condition_folder

        self.paths = [p for ext in exts for p in Path(f'{root}').glob(f'**/*.{ext}')]
        self.transform = T.Compose([T.Pad(padding=padding, fill=0),
                                    T.Resize(image_size),
                                    T.ToTensor()])
        
        if self.condition_folder is not None:
            filenames = [os.path.basename(p) for p in self.paths]
            self.condition_paths = [os.path.join(condition_folder, "images", f) for f in filenames]

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)

        condition_img = None

        if self.condition_folder is not None:
            condition_path = self.condition_paths[index]
            condition_img = Image.open(condition_path)
            condition_img = self.transform(condition_img)


        return img, condition_img, index
    

    def get_image_paths(self, indices, relative: bool = True):
        """ Get the paths of given indices. If relative is True, the relative path 
            w.r.t the dataset root is reuturned. Otherwise the dataset root is included 
            into the returned path. 
        """

        root_path = Path(self.root)
        paths = []
        for idx in indices:
            current_path = Path(self.paths[idx])

            if relative:
                current_path = os.path.relpath(current_path, root_path)

            paths.append(current_path)

        return paths


def load_diffusion_ckpt(path,
                        device: torch.device = torch.device("cpu"), 
                        ema: bool = True):

    data = torch.load(path, map_location=device)

    if not ema: return data["model"]

    # remove the prefix "model_ema." from the state_dict keys.
    model_ema = data["ema"]
    state_dict = {}

    for k, v in model_ema.items():
        if "ema_model" in k:
            new_k = k[10: ] # remove model_ema.
            state_dict[new_k] = v

    return state_dict


@torch.no_grad()
def add_noise(x, t, unwrapped_diffusion, accelerator):
    """ Add noise to images. """
    t = torch.full((len(x),), fill_value=t, device=accelerator.device).long()
    return unwrapped_diffusion.q_sample(x_start=x, t=t, noise=torch.randn_like(x))


@torch.no_grad()
def remove_noise(x, x_cond, t, unwrapped_diffusion, return_all_timesteps=False):
    """ Remove noise to images. """

    imgs = [x]

    for t_current in tqdm(reversed(range(0, t))):
        x, _ = unwrapped_diffusion.p_sample(x, t_current, x_cond=x_cond)
        imgs.append(x)

    ret = x if not return_all_timesteps else torch.stack(imgs, dim=1)

    return ret


@torch.no_grad()
def remove_noise_ddim(x,
                      x_cond,
                      t, 
                      unwrapped_diffusion, 
                      sampling_timesteps=100, 
                      return_all_timesteps=False):
    
    """ Remove noise to an image with ddim sampling. """

    batch = len(x)
    device = unwrapped_diffusion.betas.device
    total_timesteps = t 
    eta = unwrapped_diffusion.ddim_sampling_eta

    times = torch.linspace(-1, total_timesteps-1, steps=sampling_timesteps+1)   
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) 

    img = x 
    imgs = [img]

    x_start = x_cond

    for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):

        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        # self_cond = x_start if unwrapped_diffusion.self_condition else None
        self_cond = x_cond

        # print("in remove noise ddim:", self_cond.shape)
        pred_noise, x_start, *_ = unwrapped_diffusion.model_predictions(
                                                 img, 
                                                 time_cond, 
                                                 self_cond, 
                                                 clip_x_start=True, 
                                                 rederive_pred_noise=True)

        if time_next < 0:
            img = x_start
            imgs.append(img)
            continue

        alpha = unwrapped_diffusion.alphas_cumprod[time]
        alpha_next = unwrapped_diffusion.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    return ret


def load_hparms(path: str) -> dict:
    """ Loads the hparms from a yaml file and returns a dict. """
    with open(path, "r") as f: hparams = yaml.safe_load(f)
    return hparams


@torch.inference_mode()
def main(dataset_path: str, 
         out_path: str, 
         ckpt_path: str,
         config_path: str,
         noise_steps: int, 
         sampling_steps: int,
         batch_size: int,
         condition_folder: str = None,
         padding: List = [0, 0, 0, 0], # left, top, right and bottom
         remove_padding_at_the_end: bool = True):
    """
        Args:
            dataset_path (str): the dataset to be converted.
            out_path (str): the output path for the converted dataset.
            ckpt_path (str): the ckpt of the diffusion to be used.
            config_path (str): the config file of the diffusion to be used.

            noise_steps (int): the number of steps to add to every image.
            sampling_steps (int): the number of sampling steps to remove noise 
            (it should be less or equal to the number of noise steps).
            batch_size (int): the batch size for the diffusion model.
            condition_folder (str): the path to the condition folder.
            padding (list): how much to pad input images (left, top, right and bottom).
            remove_padding_at_the_end (bool): True to remove padding before saving 
            images.
    """

    
    accelerator = Accelerator(split_batches=True)

    accelerator.print("START")

    # load hparams of unet and diffusion and build the model
    hparams = load_hparms(config_path)
    model = Unet(**hparams["unet"])
    diffusion = GaussianDiffusion(model, **hparams["diffusion"])

    accelerator.print("LOADED MODEL...")

    # get image_size from hparams and load dataset to be converted
    image_size = hparams["diffusion"]["image_size"]
    dataset= ImageFolderIdx(dataset_path, 
                            image_size=image_size, 
                            padding=padding, 
                            condition_folder=condition_folder)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    accelerator.print("LOADED DATA...")

    # get state_dict from ckpt and load it into the model
    state_dict= load_diffusion_ckpt(ckpt_path, device="cpu")
    diffusion.load_state_dict(state_dict=state_dict, strict=True)

    accelerator.print("LOADED CKPT...")

    # prepare diffusion and loader with accelerator 
    # (keep a reference of unwrapped diffusion)
    diffusion, loader = accelerator.prepare(diffusion, loader)
    unwrapped_diffusion = accelerator.unwrap_model(diffusion)
    diffusion.eval()

    accelerator.print("PREPARED MODEL...")

    for images, conditions, idxs in loader:

        # normalize images
        images_normalized = unwrapped_diffusion.normalize(images)

        # 1) Add noise to images
        images_normalized = add_noise(images_normalized, 
                                          t=noise_steps, 
                                          unwrapped_diffusion=unwrapped_diffusion, 
                                          accelerator=accelerator)

        # 2.1) Remove noise in the case where the sampling steps == noise_steps
        if sampling_steps == noise_steps:
            images_normalized = remove_noise(x=images_normalized,
                                             x_cond=conditions,
                                             t=sampling_steps,
                                             diffusion=diffusion)
            
        # 2.2) Remove noise in the case where the sampling steps < noise_steps
        else:
            images_normalized = remove_noise_ddim(
                                            x=images_normalized,
                                            x_cond=conditions,
                                            t=noise_steps,
                                            unwrapped_diffusion=unwrapped_diffusion,
                                            sampling_timesteps=sampling_steps,
                                            return_all_timesteps=False)


        out_images = unwrapped_diffusion.unnormalize(images_normalized)

        out_paths = dataset.get_image_paths(indices=idxs, relative=True)

        for image, path in zip(out_images, out_paths):
            # get full path of new, converted, image
            full_path = Path(out_path)/Path(path)
            # create the folders if necessary
            Path(full_path).parent.mkdir(exist_ok=True, parents=True)
            # save image

            if remove_padding_at_the_end:
                left, top, right, bottom = padding
                _, h, w = image.shape
                image = image[:, top:h-bottom, left:w-right]

            utils.save_image(image, full_path)


if __name__ == "__main__":
    Fire(main)

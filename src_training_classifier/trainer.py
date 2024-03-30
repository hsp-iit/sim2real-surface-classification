from multiprocessing import cpu_count

# torch stuff
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

# multi gpu
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

# data
from torch.utils.data import DataLoader
from data import get_train_transform, get_test_transform, ImageFolderIdx

# model
from dann_model import DannModel, get_discriminator, GradientReversalLayer
from composed_model import ComposedModel

# optimizer and scheduler
from torch.optim import SGD
from lr_scheduler import ExponentialScheduler

# evaluator
from evaluator import eval

# paths
from pathlib import Path
from os.path import join

from tqdm import tqdm 
from math import exp

from typing import Tuple

# set seed of torch
torch.manual_seed(0)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def exists(x):
    return x is not None


def read_label_map(path: str):
    with open(path, "r") as f:
        lines = [l.rstrip().split(":") for l in f.readlines()]
    labels_map = -torch.ones([len(lines)]).long()

    for idx, label in lines:
        labels_map[int(idx)] = int(label)
    return labels_map


class Finetuner(object):

    def __init__(
        self,
        model: ComposedModel,
        data_src: str,
        data_tgt: str,
        labels_file: str,
        *,
        test_data_path: str = None,
        test_label_map_path: str = None,

        train_batch_size: int = 64,
        gradient_accumulate_every: int = 1,

        train_use_epochs: bool = False,
        train_epochs: int = 100,

        train_use_steps: bool = True, 
        train_num_steps: int = 10000,
        eval_percentage: float = 0.15, 
        eval_every: int = 400,
        
        eval_metric: str = "accuracy_macro", # or accuracy_micro
        eval_metric_stop_value: float = 0.9995, 

        # data
        resize_size: Tuple[int, int] = (320, 320),
        crop_size: Tuple[int, int] = (320, 320),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        pad: Tuple[int, int, int] = (40, 0, 40, 0),
        
        # SGD optimizer hparms
        train_lr_feature_extractor = 1e-3,
        train_lr_bottleneck = 1e-2,
        train_lr_classifier = 1e-2,
        train_lr_discriminator = 1e-2,

        weight_decay: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,

        # DISCRIMINATOR
        dis_hidden_size: int = 1024,
        dis_leaky_slope: float = 0.,
        dis_spectral_norm: bool = True,
        warm_reversal: bool = True,
        wr_high: float = 1.,
        wr_low: float = 0.,
        wr_max_steps: int = 100,
        wr_alpha: float = 1.,

        beta: float = 1, 

        # exponential lr scheduling 
        gamma: float = 10.,
        power: float = 0.75,
        
        label_smoothing: float = 0.1,
        dis_label_smoothing: float = 0.05,

        ckpt_folder = './results',
        ckpt_name = 'checkpoint.ckpt',
        fp16 = False,
        
        clip_grad_norm = -1.,
        seed = 0,

        sync_bn = True,
        **kwargs
    ):  
        
        torch.manual_seed(seed)

        assert train_use_epochs or train_use_steps
  
        super().__init__()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(split_batches=True,
                                       mixed_precision = 'fp16' if fp16 else 'no',
                                       even_batches=False,
                                       kwargs_handlers=[ddp_kwargs])

        self.num_processes = self.accelerator.num_processes
        self.global_batch_size = train_batch_size
        self.process_batch_size = train_batch_size // self.num_processes

        self.gradient_accumulate_every = gradient_accumulate_every
        self.clip_grad_norm = clip_grad_norm


        ############################ DATA ###############################################
        train_transform = get_train_transform(mean=mean, 
                                              std=std, 
                                              resize_size=resize_size, 
                                              crop_size=crop_size,
                                              pad=pad)
        
        test_transform = get_test_transform(mean=mean, 
                                            std=std, 
                                            resize_size=resize_size, 
                                            crop_size=crop_size,
                                            pad=pad)
        
        # SOURCE
        dataset_src = ImageFolderIdx(root=data_src, 
                                     transform=train_transform,
                                     csv_labels=join(data_src, labels_file))

        self.dataset_src_train, self.dataset_src_val = \
            dataset_src.split(train_per=1-eval_percentage, seed=seed)
        
        self.dataset_src_train.set_transform(train_transform)
        self.dataset_src_val.set_transform(test_transform)

        # TARGET
        self.dataset_tgt = ImageFolderIdx(root=data_tgt, transform=train_transform)

        # TEST
        self.test_dataset = None
        self.test_labels_map = None

        if test_data_path is not None:
            self.test_dataset = ImageFolderIdx(root=test_data_path, 
                                               transform=test_transform, 
                                               csv_labels=join(test_data_path, "labels.csv"))
        
        if test_label_map_path is not None:
            self.test_labels_map = read_label_map(test_label_map_path)


        self.src_loader = DataLoader(self.dataset_src_train, 
                                     batch_size=train_batch_size//2, 
                                     drop_last=True,
                                     shuffle=True, 
                                     pin_memory=True, 
                                     num_workers=cpu_count())
        
        self.tgt_loader = DataLoader(self.dataset_tgt, 
                                     batch_size=train_batch_size//2, 
                                     drop_last=True,
                                     shuffle=True, 
                                     pin_memory=True, 
                                     num_workers=cpu_count())
        

        # the evaluation metric and the value above which the training stops
        self.eval_metric = eval_metric
        self.eval_metric_stop_value = eval_metric_stop_value
        

        ################################### setup DANN ##################################
        self.beta = beta

        if warm_reversal:
            diff = wr_high - wr_low

            reversal_coeff = lambda step: \
                                        2.0 * diff / (1.0 + exp(-wr_alpha * step / \
                                        wr_max_steps)) - diff + wr_low
        else:
            reversal_coeff = None


        self.reversal = GradientReversalLayer(reversal_coeff)
        self.discriminator = get_discriminator(model=model,
                                               input_dim=model.get_features_dim(), 
                                               hidden_size=dis_hidden_size,
                                               spectral_norm=dis_spectral_norm, 
                                               leaky_slope=dis_leaky_slope)

        self.model = DannModel(model=model, 
                               reversal=self.reversal, 
                               discriminator=self.discriminator)


        if sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)



        ############################### OPTIMIZER #######################################

        param_groups = self.model.get_param_groups(lr_model=train_lr_feature_extractor, 
                                                   lr_bottleneck=train_lr_bottleneck, 
                                                   lr_classifier=train_lr_classifier,
                                                   lr_discriminator=train_lr_discriminator)
        
        self.optimizer = SGD(params=param_groups, 
                             momentum=momentum, 
                             weight_decay=weight_decay, 
                             nesterov=nesterov) # type: ignore

        
        ############################## criterion ########################################
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing,
                                          weight=torch.tensor([2.0,1.0,1.0,1.0]))
        self.criterion_dis =  BCEWithLogitsLoss()

        n_labels = train_batch_size//(2*self.num_processes)

        src_domain_labels = torch.zeros(n_labels, device=self.accelerator.device) \
                          + dis_label_smoothing
        tgt_domain_labels = torch.ones(n_labels, device=self.accelerator.device)  \
                          - dis_label_smoothing

        self.labels_domains = torch.cat([src_domain_labels, tgt_domain_labels], dim=0)
        self.labels_domains = self.labels_domains.unsqueeze(-1)


        # output directory
        self.ckpt_folder = ckpt_folder
        self.ckpt_name = ckpt_name

        if self.accelerator.is_main_process:
            Path(self.ckpt_folder).mkdir(exist_ok=True, parents=True)
        

        # step counters
        self.optimization_step = 0
        self.forward_step = 0

        # prepare stuff with accelerator

        self.model, self.optimizer, self.src_loader, self.tgt_loader, self.criterion = \
               self.accelerator.prepare(self.model, 
                                        self.optimizer, 
                                        self.src_loader,
                                        self.tgt_loader, 
                                        self.criterion)


        self.tgt_loader = cycle(self.tgt_loader)

        # total steps in one epoch
        train_steps_one_epoch = len(self.dataset_src_train) // (self.global_batch_size * self.gradient_accumulate_every)
        
        if train_use_epochs:
           train_num_steps = train_epochs * train_steps_one_epoch
           eval_every = train_steps_one_epoch

        self.train_num_steps = train_num_steps
        self.eval_every = eval_every

        self.lr_scheduler = ExponentialScheduler(optimizer=self.optimizer,
                                                 max_steps=train_num_steps,
                                                 gamma=gamma, 
                                                 power=power)
        
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)


        if train_lr_feature_extractor <= 0:
            print("Freezing backbone")
        
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            for param in unwrapped_model.model.backbone.parameters():
                param.requires_grad = False

    @property
    def device(self):
        return self.accelerator.device


    def save(self, filename: str, **kwargs):
        if not self.accelerator.is_local_main_process:
            return
        
        model = self.accelerator.unwrap_model(self.model)
        main_dict = {'step': self.optimization_step,
                     'model_dann': model.state_dict(),
                     'model': model.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict(),
                     'scaler': self.accelerator.scaler.state_dict() \
                          if exists(self.accelerator.scaler) else None}


        for k, v in kwargs.items():
            if k not in main_dict:
                main_dict[k] = v

        torch.save(main_dict, join(self.ckpt_folder, filename))

 
    def load(self, filename: str):

        data = torch.load(join(self.ckpt_folder, filename), 
                          map_location=self.accelerator.device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.optimizer.load_state_dict(data['optimizer'])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def load_model(self, filename: str):
        
        data = torch.load(join(self.ckpt_folder, filename), 
                              map_location=self.accelerator.device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=True)


    def train(self):

        accelerator = self.accelerator
        
        best_metric = -1
        early_stopped = False

        with tqdm(initial=self.optimization_step, 
                  total=self.train_num_steps, 
                  disable=not accelerator.is_main_process) as pbar:

            while (self.optimization_step < self.train_num_steps) and (not early_stopped): 
        
   
                for batch in self.src_loader:
                    
                    images_src, labels_src = batch["images"], batch["labels"]

                    images_tgt = next(self.tgt_loader)["images"]

                    images_batch = torch.cat([images_src, images_tgt], dim=0)


                    with self.accelerator.autocast():
                        model_outputs = self.model(images_batch) 

                        # get src logits
                        logits = model_outputs["logits"]
                        logits_src, logits_tgt = logits.chunk(2, dim=0)

                        # get discr logits
                        discr_logits = model_outputs["discriminator"]



                        classification_loss = self.criterion(logits_src, labels_src) / self.gradient_accumulate_every
                        
                        domain_loss = self.criterion_dis(discr_logits, self.labels_domains) / self.gradient_accumulate_every

                        loss = (classification_loss + self.beta * domain_loss) 

                        if self.optimization_step == 0 :
                            ema_loss = loss.item()
                            ema_cls_loss = classification_loss.item()
                            ema_dis_loss = domain_loss.item()
                        else:
                            ema_cls_loss = 0.95 * ema_cls_loss + 0.05 * classification_loss.item()
                            ema_dis_loss = 0.95 * ema_dis_loss + 0.05 * domain_loss.item()
                            ema_loss = 0.95 * ema_loss + 0.05 * loss.item()

                    self.accelerator.backward(loss)


                    if self.clip_grad_norm > 0:
                        accelerator.clip_grad_norm_(self.model.parameters(), 
                                                    self.clip_grad_norm)

                    pbar.set_description(f'ema_cls_loss: {ema_cls_loss:.4f} || ema_dis_loss: {ema_dis_loss:.4f}')

                    # increasing the counter for forward passes
                    self.forward_step += 1

                    if self.forward_step % self.gradient_accumulate_every == 0:
                        # we are in the case where we need to optimize
                        self.reversal.step()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        self.optimization_step += 1
                        pbar.update(1)

                    if self.optimization_step  % self.eval_every == 0:
                        eval_results = eval(dataset=self.dataset_src_val,
                                            batch_size=self.global_batch_size,
                                            model=self.model, 
                                            accelerator=self.accelerator)
                    
                        accuracy_micro = eval_results["accuracy_micro"]
                        accuracy_macro = eval_results["accuracy_macro"]

                        accelerator.print(f'EVAL: ACC_MICRO {accuracy_micro*100:.2f}%' +
                                          f' || ACC_MACRO {accuracy_macro*100:.2f}%')


                        if "micro" in self.eval_metric.lower():
                            metric = accuracy_micro
                        elif "macro" in self.eval_metric.lower():
                            metric = accuracy_macro
                        else:
                            raise ValueError("Eval_metric has a wrong value!")

                        if self.test_dataset is not None:

                            #len(self.test_dataset))
                            eval_results_test = eval(dataset=self.test_dataset,
                                                     batch_size=self.global_batch_size,
                                                     model=self.model, 
                                                     accelerator=self.accelerator,
                                                     labels_map=self.test_labels_map)
                    
                            accuracy_micro_test = eval_results_test["accuracy_micro"]
                            accuracy_macro_test = eval_results_test["accuracy_macro"]

                            accelerator.print(f'TEST: ACC_MICRO {accuracy_micro_test*100:.2f}%' +
                                            f' || ACC_MACRO {accuracy_macro_test*100:.2f}%')
                            


                        if metric > best_metric: 
                            accelerator.print(f"SAVING MODEL TO {self.ckpt_name}")
                            best_metric = metric
                            
                            self.save(filename=self.ckpt_name)

                            if metric > self.eval_metric_stop_value: 
                                accelerator.print('Training completed in advance.')
                                early_stopped = True
                                break


        accelerator.wait_for_everyone()

        accelerator.print('training complete')
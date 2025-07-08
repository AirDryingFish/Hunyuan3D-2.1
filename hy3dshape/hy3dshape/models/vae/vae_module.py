import os
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model

class VAE(pl.LightningModule):
    def __init__(
        self,
        shape_model_config,


    ):
        super().__init__()

        self.shape_model = instantiate_from_config(shape_model_config)


    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.model.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        return optimizers, schedulers
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        pass
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)


    def on_train_epoch_start(self) -> None:
        pl.seed_everything(self.trainer.global_rank)


    def forward(self, batch):
        latents, posterior = self.shape_model.encode(batch['surface'], return_kl=True)
        points = batch['geo_points']
        target = batch['geo_labels']
        num = self.cfg.n
        coarse_target = target[:,:num]
        sharp_target = target[:,num:]
        criteria = torch.nn.MSELoss()

        loss_kl = posterior.kl()
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

        latents = self.shape_model.decode(latents)
        logits = self.shape_model.geo_decoder(queries=points,latents=latents)

        coarse_logits = logits[:,:num]
        sharp_logits = logits[:,num:]

        return {
            "loss_coarse": criteria(coarse_logits, coarse_target).mean(),
            "loss_sharp": criteria(sharp_logits, sharp_target).mean(),
            "loss_kl": loss_kl,
            "overall_logits": logits,
            "overall_target": target,
            "coarse_logits": coarse_logits,
            "coarse_target": coarse_target,
            "sharp_logits": sharp_logits,
            "sharp_target": sharp_target,
        }

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
        loss = loss['loss_coarse'] + loss['loss_sharp'] + loss['loss_kl']
        split = 'train'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        out = self.forward(batch)
        split = 'val'

        threshold = 0
        overall_outputs = out["overall_logits"]
        overall_labels = out["overall_target"]
        overall_labels = (overall_labels >= threshold).float()
        overall_pred = torch.zeros_like(overall_outputs)
        overall_pred[overall_outputs>=threshold] = 1
        overall_accuracy = (overall_pred==overall_labels).float().sum(dim=1) / overall_labels.shape[1]
        overall_accuracy = overall_accuracy.mean()
        overall_intersection = (overall_pred * overall_labels).sum(dim=1)
        overall_union = (overall_pred + overall_labels).gt(0).sum(dim=1)
        overall_iou = overall_intersection * 1.0 / overall_union + 1e-5
        overall_iou = overall_iou.mean()

        coarse_outputs = out["coarse_logits"]
        coarse_labels = out["coarse_target"]
        coarse_labels = (coarse_labels >= threshold).float()
        coarse_pred = torch.zeros_like(coarse_outputs)
        coarse_pred[coarse_outputs>=threshold] = 1
        coarse_accuracy = (coarse_pred==coarse_labels).float().sum(dim=1) / coarse_labels.shape[1]
        coarse_accuracy = coarse_accuracy.mean()
        coarse_intersection = (coarse_pred * coarse_labels).sum(dim=1)
        coarse_union = (coarse_pred + coarse_labels).gt(0).sum(dim=1)
        coarse_iou = coarse_intersection * 1.0 / coarse_union + 1e-5
        coarse_iou = coarse_iou.mean()

        sharp_outputs = out["sharp_logits"]
        sharp_labels = out["sharp_target"]
        sharp_labels = (sharp_labels >= threshold).float()
        sharp_pred = torch.zeros_like(sharp_outputs)
        sharp_pred[sharp_outputs>=threshold] = 1
        sharp_accuracy = (sharp_pred==sharp_labels).float().sum(dim=1) / sharp_labels.shape[1]
        sharp_accuracy = sharp_accuracy.mean()
        sharp_intersection = (sharp_pred * sharp_labels).sum(dim=1)
        sharp_union = (sharp_pred + sharp_labels).gt(0).sum(dim=1)
        sharp_iou = sharp_intersection * 1.0 / sharp_union + 1e-5
        sharp_iou = sharp_iou.mean()

        loss_dict = {
            f"{split}/overall_accuracy": overall_accuracy,
            f"{split}/overall_iou": overall_iou,
            f"{split}/coarse_accuracy": coarse_accuracy,
            f"{split}/coarse_iou": coarse_iou,
            f"{split}/sharp_accuracy": sharp_accuracy,
            f"{split}/sharp_iou": sharp_iou,
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        return 



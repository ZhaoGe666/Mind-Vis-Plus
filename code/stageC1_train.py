
import os
import torch
import pytorch_lightning as PL
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from pdnorm_model import PDfLDM
from pdnorm_dataset import GOD_dataset
from omegaconf import OmegaConf


config = OmegaConf.load('./code/stageC_config.yaml')
model = PDfLDM(**config.model.params)
checkpoint_path = './results/generation/07-09-2023-01-50-29/checkpoint_best.pth'
# checkpoint_path = './pretrains/ldm/label2img/model.ckpt'  
# from the very first ckpt, the cond_stage_model should be loaded as well!
model_meta = torch.load(checkpoint_path, map_location='cpu')
model.state_from_fLDM(model_meta['model_state_dict'])
# unfreezed_params = model.unfreeze_stageC_params(True) # True-->28.2M; False-->26.2M

output_root = config.model.params.output_root  # '/data/xiaozhaoliu/stageC1'
os.makedirs(output_root,exist_ok=True)
wandb_logger = WandbLogger(project='mind-vis',
                           group='stageC1',
                           log_model=False,
                           save_dir=output_root)

trainer = PL.Trainer(accelerator='gpu',
                     devices=[3,4,5,6,7],
                     strategy='ddp', 
                     logger=wandb_logger,
                     check_val_every_n_epoch=5,
                     max_epochs=500,
                     precision=32,
                     accumulate_grad_batches=1,
                     gradient_clip_val=0.5,
                     enable_model_summary=False,
                     enable_checkpointing=True
                    )

train_set = GOD_dataset(subset='train')
valid_set = GOD_dataset(subset='valid')
train_loader = DataLoader(train_set, batch_size=15, num_workers=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=50, num_workers=64, shuffle=False)

trainer.fit(model, train_loader, valid_loader)


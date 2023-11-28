
import os
import torch
import pytorch_lightning as PL
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from pdnorm_model import PDfLDM
from pdnorm_dataset import GOD_dataset
from omegaconf import OmegaConf


def main():
    train_set = GOD_dataset(subset='train')
    valid_set = GOD_dataset(subset='valid')
    train_loader = DataLoader(train_set, batch_size=15, num_workers=64, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=50, num_workers=64, shuffle=False)

    config = OmegaConf.load('./code/stageC_config.yaml')
    group_dir = config.model.params.output_root  # '/data/xiaozhaoliu/stageC1'
    os.makedirs(group_dir,exist_ok=True)
    
    model = PDfLDM(**config.model.params)

    wandb_logger = WandbLogger(project='mind-vis',   
                            group='stageC1',                        
                            id='xhq7ymkr',
                            resume=True,
                            log_model=False,
                            save_dir=group_dir,
                            )

    trainer = PL.Trainer(accelerator='gpu',
                        devices=[3,4,5,6,7],
                        strategy='ddp',
                        resume_from_checkpoint='/data/xiaozhaoliu/stageC1/mind-vis/xhq7ymkr/checkpoints/epoch=1999-step=160000.ckpt', 
                        logger=wandb_logger,
                        check_val_every_n_epoch=5,
                        max_epochs=2500,
                        precision=32,
                        accumulate_grad_batches=1,
                        gradient_clip_val=0.5,
                        enable_model_summary=False,
                        enable_checkpointing=True
                        )

    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    main()

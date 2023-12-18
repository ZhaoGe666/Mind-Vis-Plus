
import os
import torch
import pytorch_lightning as PL
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from pdnorm_model import PDfLDM
from pdnorm_dataset import GOD_dataset



def main():
    '''
    before running this script, always check these 6 variables:
      - subjects
      - group_dir
      - id
      - devices
      - resume_from_checkpoint
      - max_epochs
    '''
    train_set = GOD_dataset(subset='train',subjects=['sbj_3'])
    valid_set = GOD_dataset(subset='valid',subjects=['sbj_3'])
    train_loader = DataLoader(train_set, batch_size=15, num_workers=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=50, num_workers=32, shuffle=False)

    config = OmegaConf.load('./code/stageC_config.yaml')
    group_dir = config.model.params.output_root  # '/data/xiaozhaoliu/stageC1'
    os.makedirs(group_dir,exist_ok=True)
    
    model = PDfLDM(**config.model.params)

    wandb_logger = WandbLogger(project='mind-vis',   
                            group='stageC1',                        
                            id='ihwhs3ps',
                            resume=True,
                            log_model=False,
                            save_dir=group_dir,
                            )
    
    checkpoint_callback = ModelCheckpoint(save_top_k=3, 
                                          monitor='val/top-1-class',
                                          mode='max')

    trainer = PL.Trainer(accelerator='gpu',
                        devices=[3,4,5,6,7],
                        strategy='ddp',
                        resume_from_checkpoint='/home/xiaozhaoliu/Mind-Vis-Plus/results/stageC1/mind-vis/ihwhs3ps/checkpoints/epoch=499-step=8000.ckpt', 
                        logger=wandb_logger,
                        check_val_every_n_epoch=5,
                        max_epochs=1000,
                        precision=32,
                        accumulate_grad_batches=1,
                        gradient_clip_val=0.5,
                        enable_model_summary=False,
                        enable_checkpointing=True,
                        callbacks=[checkpoint_callback]
                        )


    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    main()

import os
from pytz import timezone
from datetime import datetime
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import pytorch_lightning as PL
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from pdnorm_model import PDfLDM
from pdnorm_dataset import GOD_dataset


def main():
    '''
    before running this script, always check these 5 variables:
      - subjects
      - group_dir
      - fmri_encoder_ckpt_path
      - devices
      - max_epochs
    '''
    train_set = GOD_dataset(subset='train')
    valid_set = GOD_dataset(subset='valid')  # FIXME
    train_loader = DataLoader(train_set, batch_size=15, num_workers=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=50, num_workers=32, shuffle=False)

    config = OmegaConf.load('./code/stageC_config.yaml')

    group_dir = config.model.params.output_root  # FIXME '/data/xiaozhaoliu/stageC1'
    os.makedirs(group_dir,exist_ok=True)

    model = PDfLDM(**config.model.params)
    # mae_ckpt_path = '/data/xiaozhaoliu/stageA3/09-11-2023-08-45-22/checkpoints/MAEforFMRI_epoch99.ckpt'
    # mae_meta = torch.load(mae_ckpt_path, map_location='cpu')  # from A1 or A2
    # model.state_from_mae(mae_meta['model'])
    fmri_encoder_ckpt_path = '/home/xiaozhaoliu/Mind-Vis-Plus/pretrains/GOD/fmri_encoder.pth'   # FIXME
    fmri_encoder_meta = torch.load(fmri_encoder_ckpt_path, map_location='cpu')  # from A1 or A2
    model.state_from_fmri_encoder(fmri_encoder_meta['model'])

    # from the very first ckpt, the cond_stage_model should be loaded as well!
    LDM_ckpt_path = './pretrains/ldm/label2img/model.ckpt'  
    LDM_meta = torch.load(LDM_ckpt_path, map_location='cpu')
    model.state_from_LDM(LDM_meta['state_dict'])

    # unfreezed_params = model.unfreeze_stageC_params(True) # True-->28.2M; False-->26.2M

    wandb_logger = WandbLogger(project='mind-vis',
                            group='stageC1',
                            log_model=False,
                            save_dir=group_dir, # save logs to group_dir/wandb
                            )  
    checkpoint_callback = ModelCheckpoint(save_top_k=3, 
                                          monitor='val/top-1-class',
                                          mode='max')
    
    trainer = PL.Trainer(accelerator='gpu',
                        devices=[3,4,5,6,7],    # FIXME
                        strategy='ddp', 
                        logger=wandb_logger,
                        check_val_every_n_epoch=5,
                        max_epochs=1000,   # FIXME
                        precision=32,
                        accumulate_grad_batches=1,
                        gradient_clip_val=0.5,
                        enable_model_summary=False,
                        enable_checkpointing=True,
                        callbacks=[checkpoint_callback]
                        )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()


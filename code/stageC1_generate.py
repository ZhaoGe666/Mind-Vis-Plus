
import os
import torch
import pytorch_lightning as PL
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from pdnorm_model import PDfLDM
from pdnorm_dataset import GOD_dataset
from omegaconf import OmegaConf

def main():
    # device = torch.device('cuda:7')
    config = OmegaConf.load('./code/stageC_config.yaml')
    model = PDfLDM(**config.model.params)
    checkpoint_path = './results/stageC1/mind-vis/ihwhs3ps/checkpoints/epoch=829-step=13280.ckpt'  # FIXME watch out !!
    # checkpoint_path = './pretrains/ldm/label2img/model.ckpt'  
    # from the very first ckpt, the cond_stage_model should be loaded as well!
    model_meta = torch.load(checkpoint_path, map_location='cpu')
    model.state_from_fLDM(model_meta['state_dict'])
    # unfreezed_params = model.unfreeze_stageC_params(True) # True-->28.2M; False-->26.2M

    output_root = config.model.params.output_root  # '/data/xiaozhaoliu/stageC1'
    os.makedirs(output_root,exist_ok=True)
    wandb_logger = WandbLogger(project='mind-vis',
                            id='ihwhs3ps',  # FIXME watch out !!
                            group='stageC1',
                            log_model=False,
                            save_dir=output_root,
                            prefix='test_best_829epoch',  # FIXME watch out !!
                            resume=True
                            )

    trainer = PL.Trainer(accelerator='gpu',
                        devices=[1],  # FIXME watch out !!
                        logger=wandb_logger,
                        max_epochs=1,
                        enable_progress_bar=True)

    valid_set = GOD_dataset(subset='valid', return_more_class_info=True)
    valid_loader = DataLoader(valid_set, batch_size=50, num_workers=32, shuffle=False)
    setattr(model, 'subset', 'valid')
    trainer.test(model, valid_loader)

    train_set = GOD_dataset(subset='train', return_more_class_info=True)
    train_loader = DataLoader(train_set, batch_size=50, num_workers=32, shuffle=False)
    setattr(model, 'subset', 'train')
    trainer.test(model, train_loader)


if __name__ == '__main__':
    main()

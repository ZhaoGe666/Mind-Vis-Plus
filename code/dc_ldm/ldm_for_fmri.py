import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_fmri import fmri_encoder

def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels, cond_dim=1280, global_pool=True):
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)  # 初始化一个fmri_encoder对象
        # 不从config载入 norm_layer

        model.load_checkpoint(metafile['model'])  # fmri_encoder对象的参数通过metafile加载
        # 问题是metafile哪来的，stageA也没用到fmri_encoder
        # 破案了，stageA用到的 MAEforFMRI 的state可以用于fmri_encoder，encoder部分是一样的，变量名定义也一样所以可以load_state
        self.mae = model
        self.fmri_seq_len = model.num_patches  # 291  (291*169=4656)=num_voxels
        self.fmri_latent_dim = model.embed_dim  # 1024
        if global_pool == False: # False,   load_checkpoint中，为True时，不载入LayerNorm的参数
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True), # kernel_size=1, 相当于1*1卷积
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn) # 将seq_len作为channel去reduce？意义不明
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out

class fLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.config_path = os.path.join(pretrain_root, 'config.yaml') 
        config = OmegaConf.load(self.config_path)  # LDM的默认参数
        config.model.params.unet_config.params.use_time_cond = use_time_cond  # time step conditioning
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim  # 只传入cond_stage_model

        model = instantiate_from_config(config.model)  # 根据config.yaml的超参初始化一个LatentDiffusion
        # model.model = DiffusionWrapper  ----> 未冻结
        # model.first_stage_mode = VQModelInterface; ----> .eavl()
        # model.cond_stage_model = ClassEmbedder  ----> 未冻结参数
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']  # load 预训练 Latent Diffusion
       
        m, u = model.load_state_dict(pl_sd, strict=False)  # missing keys & unexpected keys
        model.cond_stage_trainable = True  # LDM有 self.cond_stage_trainable 因此不需要再次定义
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool)

        # model.cond_stage_model.mae 为一个 fmri_encoder 对象, 还有其他部分！！
        model.ddim_steps = ddim_steps  # ddpm中有, 不需要再次定义
        model.re_init_ema() # ddpm有,需要在新的PDfLDM中再次调用，因为参数self.model改变
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels 
        # 要利用LDM中的generate方法，需要(在ldm中)重新定义self.p_channels=self.channels？
        # FIXME：本身就有，为何不是更改generate(重写的)调用时的参数？
        model.p_image_size = config.model.params.image_size
        # FIXME: 同上,存在 self.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult
        # 调用同上,但参数确实需要更新 self.chult = kwargs[][]...

        self.device = device  # TODO:与lightning对齐
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None  # shit1
        config.logger = None  # shit2
        #config.py里没有这两个参数,也不会作为参数传入任何方法

        # TODO： 以下属性需要在新model中
        self.model.main_config = config  # DDPM 里初始化为None，此处传入实例变量，在full_validation方法中用于保存checkpoint
        
        # FIXME: 由于full_validation方法中保存的是self.main_config，
        # 因此以下配置均不会更新到self.main_config中，包括batch_size,lr等
        self.model.output_path = output_path  # 
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15  # DDPM 里初始化为0.0，此处传入作用于validation_step方法
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=15, shuffle=True) 
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        # 默认 first_stage （image）是 freeze 的，cond_stage （fmri）是 trainable 的
        self.model.unfreeze_whole_model()  # 设置LatentDiffusion所有的参数requires_grad=True

        self.model.freeze_first_stage()  # 冻结 first_stage 
        #   --------->>> 最后和默认设置一样，只 train condition (fmri) 部分

        self.model.learning_rate = lr1  # FIXME：调用fitune时传入lr1的就是config.lr，闹呢？
        self.model.train_cond_stage_only = True  # FIXME: Very important but shit setting！！！
        # LDM初始化时设为False，
        # 影响LDM的configure_optimizers方法，会只选部分参数给optimizer更新(而不是设置requires_grad)
        # 将修改参数是否可训练放到方法中，在脚本中修改训练参数，而不是
        self.model.eval_avg = config.eval_avg # FIXME：init时就传入了，闹呢？
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)
        # dataloader 中 每个item形状为 {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}
        # 因此需要在 self.model (LatentDiffusion) 的 forward 中拆分输入
        self.model.unfreeze_whole_model() # 何苦呢？
        
        # 这里保存最后一个epoch的dict? config内容也更多，
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None):
        # TODO: 用此更新LDM中的generate
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['fmri']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')  # FIXME: why?
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)





import os
import time
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import ViT_L_16_Weights, vit_l_16

from tqdm import tqdm
from PIL import Image
from einops import rearrange, repeat
from abc import ABC, abstractmethod

from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

# from dc_ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from dc_ldm.util import default
from dc_ldm.models.diffusion.ddpm import LatentDiffusion, DiffusionWrapper
# from dc_ldm.models.diffusion.plms import PLMSSampler
from dc_ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from torch.nn.modules.normalization import _shape_t


class PDNorm(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def init_mlp(self):
        # force the initial values of (scale,shift) to be ~(1,0) 
        pass
    
    @abstractmethod
    def forward(self, x, prompt):
        pass


class PDLayerNorm(PDNorm, nn.LayerNorm):
    '''
    当输入为(B,C,L),normalized_shape=(L)时, 与nn.LayerNorm的区别在于:
        scale和shift在样本内的C上共享,即与样本有关
        而weight和bias在batch内的C上共享
    '''
    def __init__(self, normalized_shape: _shape_t, prompt_dim: int=16, eps: float = 0.00001, 
                 elementwise_affine: bool = False,  # False --> super.__init__中取消初始化self.weight和self.bias
                 device=None, dtype=None) -> None:
        nn.LayerNorm.__init__(self, normalized_shape, eps, elementwise_affine, device, dtype)
        # 父类中会将self.normalized_shape包装成tuple
        if len(self.normalized_shape) != 1:
            raise NotImplementedError # TODO
        self.mlp_scale = nn.Sequential(nn.Linear(prompt_dim, self.normalized_shape[0]), nn.SiLU())
        self.mlp_shift = nn.Sequential(nn.Linear(prompt_dim, self.normalized_shape[0]), nn.SiLU())
        self.init_mlp()
        # del self.weight
        # del self.bias
    def init_mlp(self):
        # force the initial (scale,shift) to be (0,0) 
        nn.init.zeros_(self.mlp_scale[0].weight)
        nn.init.zeros_(self.mlp_scale[0].bias) 
        nn.init.zeros_(self.mlp_shift[0].weight)
        nn.init.zeros_(self.mlp_shift[0].bias) 

    def forward(self, x: Tensor, prompt:Tensor) -> Tensor:
        # x:(B,291,1024)
        ############ mean和var是normalized_shape计算，也就是跨其余维度分别计算############
        mean = torch.mean(x, -1, keepdim=True)  # (B,291,1)
        var = torch.var(x, -1, keepdim=True, unbiased=False)
        x_standard = (x-mean)/torch.sqrt(var+self.eps)  # (B,291,1024)
        ############ 而 γ 和 β 是与normalized_shape同纬度，也就是跨其余维度share############
        scale = self.mlp_scale(prompt).unsqueeze(1)  #  (B,1,1024)
        shift = self.mlp_shift(prompt).unsqueeze(1)  # same   
        return x_standard * (1 + scale) + shift   
    
class PDGroupNorm(PDNorm, nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, prompt_dim: int=16, eps: float = 0.00001, 
                 affine: bool = False,  # False --> super.__init__中取消初始化self.weight和self.bias
                 device=None, dtype=None) -> None:
        nn.GroupNorm.__init__(self, num_groups, num_channels, eps, affine, device, dtype)

        self.mlp_scale = nn.Sequential(nn.Linear(prompt_dim, num_channels), nn.SiLU())
        self.mlp_shift = nn.Sequential(nn.Linear(prompt_dim, num_channels), nn.SiLU())
        self.init_mlp()

    def init_mlp(self):
        # force the initial (scale,shift) to be (0,0) 
        nn.init.zeros_(self.mlp_scale[0].weight) 
        nn.init.zeros_(self.mlp_scale[0].bias) 
        nn.init.zeros_(self.mlp_shift[0].weight) 
        nn.init.zeros_(self.mlp_shift[0].bias) 

    def forward(self, x: Tensor, prompt:Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B, self.num_groups, C//self.num_groups, H, W)  # Reshape for group operations
        mean = torch.mean(x_reshaped, dim=(2, 3, 4), keepdim=True)
        var = torch.var(x_reshaped, dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_standard = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        x_standard = x_standard.reshape(B, C, H, W)  # Reshape back
        
        scale = self.mlp_scale(prompt).unsqueeze(-1).unsqueeze(-1)
        shift = self.mlp_shift(prompt).unsqueeze(-1).unsqueeze(-1)

        return x_standard * (1 + scale) + shift


# class PDLayerNorm(PDNorm, nn.LayerNorm):
#     '''
#     当输入为(B,C,L),normalized_shape=(L)时, 与nn.LayerNorm的区别在于:
#         scale和shift在样本内的C上共享,即与样本有关
#         而weight和bias在batch内的C上共享
#     '''
#     def __init__(self, normalized_shape: _shape_t, prompt_dim: int=16, eps: float = 0.00001, 
#                  elementwise_affine: bool = False,  # False --> super.__init__中取消初始化self.weight和self.bias
#                  device=None, dtype=None) -> None:
#         nn.LayerNorm.__init__(self, normalized_shape, eps, elementwise_affine, device, dtype)
#         # 父类中会将self.normalized_shape包装成tuple
#         if len(self.normalized_shape) != 1:
#             raise NotImplementedError # TODO
#         self.mlp_scale = nn.Sequential(nn.Linear(prompt_dim, self.normalized_shape[0]), nn.SiLU())
#         self.mlp_shift = nn.Sequential(nn.Linear(prompt_dim, self.normalized_shape[0]), nn.SiLU())
#         self.init_mlp()
#         # del self.weight
#         # del self.bias
#     def init_mlp(self):
#         # force the initial (scale,shift) to be (1,0) 
#         nn.init.normal_(self.mlp_scale[0].weight,mean=0,std=0.001)  # # around 0
#         nn.init.ones_(self.mlp_scale[0].bias)  # around 1
#         nn.init.normal_(self.mlp_shift[0].weight,mean=0,std=0.001)  # around 0
#         nn.init.zeros_(self.mlp_shift[0].bias)  # around 0

#     def forward(self, x: Tensor, prompt:Tensor) -> Tensor:
#         # x:(B,291,1024)
#         ############ mean和var是normalized_shape计算，也就是跨其余维度分别计算############
#         mean = torch.mean(x, -1, keepdim=True)  # (B,291,1)
#         var = torch.var(x, -1, keepdim=True, unbiased=False)
#         x_standard = (x-mean)/torch.sqrt(var+self.eps)  # (B,291,1024)
#         ############ 而 γ 和 β 是与normalized_shape同纬度，也就是跨其余维度share############
#         scale = self.mlp_scale(prompt).unsqueeze(1)  #  (B,1,1024)
#         shift = self.mlp_shift(prompt).unsqueeze(1)  # same
#         out = x_standard * scale + shift
#         # out = F.layer_norm(
#         #     x, self.normalized_shape, scale, shift, self.eps)        
#         return out


# class PDGroupNorm(PDNorm, nn.GroupNorm):
#     def __init__(self, num_groups: int, num_channels: int, prompt_dim: int=16, eps: float = 0.00001, 
#                  affine: bool = False,  # False --> super.__init__中取消初始化self.weight和self.bias
#                  device=None, dtype=None) -> None:
#         nn.GroupNorm.__init__(self, num_groups, num_channels, eps, affine, device, dtype)

#         self.mlp_scale = nn.Sequential(nn.Linear(prompt_dim, num_channels), nn.SiLU())
#         self.mlp_shift = nn.Sequential(nn.Linear(prompt_dim, num_channels), nn.SiLU())
#         self.init_mlp()

#     def init_mlp(self):
#         # force the initial (scale,shift) to be (1,0) 
#         nn.init.normal_(self.mlp_scale[0].weight,mean=0,std=0.001)  # # around 0
#         nn.init.ones_(self.mlp_scale[0].bias)  # around 1
#         nn.init.normal_(self.mlp_shift[0].weight,mean=0,std=0.001)  # around 0
#         nn.init.zeros_(self.mlp_shift[0].bias)  # around 0

#     def forward(self, x: Tensor, prompt:Tensor) -> Tensor:
#         B, C, H, W = x.shape
#         x_reshaped = x.reshape(B, self.num_groups, C//self.num_groups, H, W)  # Reshape for group operations
#         mean = torch.mean(x_reshaped, dim=(2, 3, 4), keepdim=True)
#         var = torch.var(x_reshaped, dim=(2, 3, 4), keepdim=True, unbiased=False)
#         x_standard = (x_reshaped - mean) / torch.sqrt(var + self.eps)
#         x_standard = x_standard.reshape(B, C, H, W)  # Reshape back
        
#         scale = self.mlp_scale(prompt).unsqueeze(-1).unsqueeze(-1)
#         shift = self.mlp_shift(prompt).unsqueeze(-1).unsqueeze(-1)

#         return x_standard * scale + shift
    


###########################################################################
###########################################################################


class PDfLDM(LatentDiffusion):  # 200 Norm instances in
    
    
    # TODO:将fLDM的方法如finetune和generate直接并入, 包括 __init__!!!
    def __init__(self, first_stage_config, cond_stage_config, 
                 output_root, run_full_validation_threshold, eval_avg, learning_rate, global_pool,
                 image_size, channels,
                 enable_multi_subject=True, enable_multi_dataset=False, datasets=['GOD'], subjects = [],
                 num_timesteps_cond=None, cond_stage_key='fmri', cond_stage_trainable=True, 
                 concat_mode=True, cond_stage_forward=None, conditioning_key=None, 
                 scale_factor=1, scale_by_std=False, *args, **kwargs):
        # print('\U0001F642'*2, f' {self.__class__.__name__}: initializing...', '\U0001F642'*2)
        super().__init__(first_stage_config, cond_stage_config, 
                         num_timesteps_cond, cond_stage_key, cond_stage_trainable, 
                         concat_mode, cond_stage_forward, conditioning_key, 
                         scale_factor, scale_by_std, *args, **kwargs)

        self.__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}
        self.enable_multi_dataset = enable_multi_dataset
        self.enable_multi_subject = enable_multi_subject
        self.global_pool = global_pool

        if self.enable_multi_dataset:
            # assert (datasets == ['GOD','BLOD5000','NSD']) or (datasets == ['GOD','BLOD5000']), ...
            assert datasets == ['GOD','BLOD5000','NSD'], ...
            f'启用了multi_dataset, 但传入模型的 dataset 为: {datasets}' 
            self.embed_dataset = nn.Embedding(num_embeddings=3, embedding_dim=8)
            self.embed_subject = nn.Embedding(num_embeddings=50, embedding_dim=16)  # TODO：num_embeddings 设置问题
            self.prompt_embedding_dim = 24
        else:
            assert datasets == ['GOD']  # subjects = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5']
            self.embed_subject = nn.Embedding(num_embeddings=5, embedding_dim=16)  # num_embeddings 可以大于5吗？
            self.prompt_embedding_dim = 16

        self.model = PDDiffusionWrapper(kwargs['unet_config'], conditioning_key,
                                        prompt_dim=self.prompt_embedding_dim, global_pool=self.global_pool)  # 109 Norm instances in
        # overrided from: dc_ldm.models.diffusion.ddpm.DiffusionWrapper

        # self.first_stage_model = VQModelInterface()  # from: dc_ldm.models.autoencoder.VQModelInterface
        #  42 Norm instances in, but not trainable, so there is no need to override this pre-trained module 

        self.cond_stage_model = PDCondStageModel(prompt_dim=self.prompt_embedding_dim, global_pool=self.global_pool,
                                                 **cond_stage_config['params'])  # 49 Norm instances in
        # overrided! configs are initialized with default values from a stageB checkpoint (the conf)
        # if logger is not None:
        #     logger.watch(model, log="all", log_graph=False)
        self.ch_mult = first_stage_config['params']['ddconfig']['ch_mult']
        # self.main_config 不需要保存config
        # TODO：后续所有(有调整必要的)超参应该放入config文件管理，与checkpoint分开管理
        
        # self.output_path = output_root + '/' + \
        #     datetime.utcnow().astimezone(timezone('Asia/Shanghai')).strftime("%y-%m-%d-%H-%M-%S")
        self.run_full_validation_threshold = run_full_validation_threshold
        self.eval_avg = eval_avg
        self.learning_rate = learning_rate
        # self.full_val_interval = full_val_interval
        # self.n_random_sampling_minor = n_random_sampling_minor
        # self.n_random_sampling_major = n_random_sampling_major
        # self.n_partial_sampling = n_partial_sampling
        self.image_size = image_size
        self.channels = channels

        self.re_init_ema()
        # 默认冻结first_stage_model参数，其余均放开
        self.unfreeze_whole_model()
        self.freeze_first_stage()
        # self.train_cond_stage_only = True # 设置标志位，见configure_optimizers
        # print('\U0001F642'*2, f' {self.__class__.__name__}: initialized!', '\U0001F642'*2)
        # self.count_trainable_parameters()

    
    def count_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print('-' * 48)
        print(f"{self.__class__.__name__}: {(trainable_params+non_trainable_params)/(1024*1024):>32.1f} M")
        print(f"      Trainable     params: {trainable_params/(1024*1024):>12.1f} M")
        print(f"      Non-trainable params: {non_trainable_params/(1024*1024):>12.1f} M")
        print('=' * 48)
        for name, module in self.named_children():
            if name != 'embed_subject':
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                non_trainable_params = sum(p.numel() for p in module.parameters() if not p.requires_grad)
                print(f"  Module: {module.__class__.__name__}")
                print(f"      Trainable     params: {trainable_params/(1024*1024):>12.1f} M")
                print(f"      Non-trainable params: {non_trainable_params/(1024*1024):>12.1f} M")
                print('-' * 48)

    # overrided
    def state_from_fLDM(self, original_state_dict): 
        missing_keys, unexpected_keys = self.load_state_dict(original_state_dict, strict=False)
        # print('Missing Keys:')
        for mk in missing_keys:
            assert 'mlp_' in mk or 'decoder' in mk or 'embed_subject' in mk, f'\U0001F60E missing param: {mk}'
        print('\U0001F60E'*2 + f' {self.__class__.__name__}: State dict of fLDM (from stageB) successfully loaded! '+'\U0001F60E'*2)


    def state_from_fmri_encoder(self, original_state_dict): 
        del original_state_dict['mask_token']
        del original_state_dict['pos_embed']
        del original_state_dict['decoder_pos_embed']
        missing_keys, unexpected_keys = self.cond_stage_model.mae.load_state_dict(original_state_dict, strict=False)
        # print('Missing Keys:')
        for uk in unexpected_keys: 
            assert 'decoder' in uk or 'norm' in uk, f'\U0001F605 unexpected param: {uk} '
        print('\U0001F605'*2+' State dict of fmri encoder (from stage A) is successfully loaded! '+'\U0001F605'*2)

    def state_from_LDM(self, original_state_dict): 
        missing_keys, unexpected_keys = self.load_state_dict(original_state_dict, strict=False)
        # print('Missing Keys:')
        for mk in missing_keys:  # 先使用
            assert 'mlp_' in mk or \
                'cond_stage_model' in mk or \
                'embed_subject' in mk or \
                'time_embed_condtion' in mk, f'\U0001F60F missing param: {mk}'
        print('\U0001F60F'*2+' State dict of LDM is successfully loaded! '+'\U0001F60F'*2)
    
    def unfreeze_stageC_params(self):
        # unfreeze the 

        cond_params = list(self.cond_stage_model.parameters())  # 502 个参数，包含49个norm
        unet_cond_params = [p for n, p in self.model.named_parameters() 
                    if 'time_embed_condtion' in n or 'attn2' in n]   # Unet 150/ self 300
        # 'time_embed_condtion'  () UNetModel.Sequential(conv_[, linear])
        # 'attn2'                () UNetModel.SpatialTransformer.BasicTransformerBlocker.CrossAttention
        # 'norm2'                () UNetModel.SpatialTransformer.BasicTransformerBlocker.Layernorm
            # 除了以上两部分参数包含的pdnorm，将外部的pdnorm也放开训练
            # 已知cond_stage_model.mae.decoder_blocks和first_stage_model中没有pdnorm，因此额外的pdnorm只存在于unet
            # num_instances_pdnorm: 158 = 49(cond_stage_model) + 109(unet_model)
        unet_pdnorm_params = []
        for name, module in self.model.named_modules():
            if isinstance(module, PDNorm):
                unet_pdnorm_params.extend(list(module.parameters()))
        prompt_embed_params = list(self.embed_subject.parameters())
        if self.enable_multi_dataset:
            prompt_embed_params.extend(list(self.embed_dataset.parameters()))
        unfreezed_params = cond_params + unet_cond_params + unet_pdnorm_params + prompt_embed_params

        self.freeze_whole_model()
        for p in unfreezed_params: # save VRAM when computing gradients
                p.requires_grad = True
        if self.global_rank == 0:
            self.count_trainable_parameters()
        return unfreezed_params

    def configure_optimizers(self):

        # if self.train_cond_stage_only = True
        unfreezed_params = self.unfreeze_stageC_params()
        if self.global_rank == 0:
            print(f'Rank{self.global_rank}: Unfreezing {len(unfreezed_params)} tensors and setting up the optimizer...\n',
                'Only optimize parameters from:\n',
                '     (1) cond_stage_model (whole)\n',
                '     (2) unet_model (cross_attention_heads, projection_heads and all pdnorms)\n',
                '     (3) prompt_embed')
        opt = torch.optim.AdamW(unfreezed_params, lr=self.learning_rate)

        # if self.use_scheduler:
        #     assert 'target' in self.scheduler_config
        #     scheduler = instantiate_from_config(self.scheduler_config)

        #     print("Setting up LambdaLR scheduler...")
        #     scheduler = [
        #         {
        #             'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
        #             'interval': 'step',
        #             'frequency': 1
        #         }]
        #     return [opt], scheduler
        return opt

    def apply_model(self, x_noisy, prompt, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}  # {'c_crossattn': c}  c为经cond_stage_model.encode的fmri latent

        x_recon = self.model(x_noisy, prompt, t, **cond)  # 使用DiffusionWrapper(实际为UNetModel)重建图像


        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    # overrided
    def p_losses(self, x_start, cond, t, prompt, noise=None):
        #  x_start:(B,3,64,64)  cond:(B,291,1024)  t:(B)
        noise = default(noise, lambda: torch.randn_like(x_start))  # (B,3,64,64)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 某个t时刻的加噪图像
        model_output = self.apply_model(x_noisy, prompt, t, cond)  
        # train时，给定x_noisy,p,c重建1个t时刻的噪音
        # 而要求valid时给random_noise, p, 能重建任意时刻的噪音。用随机噪音减去后即可得到z0？
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))  # shape = (B)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
    

    def conditioning_forward(self, c, p):
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c, p)  #  这里的encode应该放到cond_stage_model而不是MAE
        else:
            raise NotImplementedError
        return c
        
    def forward(self, z, p, c, *args, **kwargs):
        # p = prompt_embedding : (B,16)
        #  z:(B,3,64,64)  c:(B,4656) self.cond_stage_trainable==True
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long() # t.shape = bs, t.dim()=1
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable: # 在 True
                c = self.conditioning_forward(c, p)  # c:(B,291,1024)
                if self.return_cond:
                    return self.p_losses(z, c, t, p, *args, **kwargs), c
        return self.p_losses(z, c, t, p, *args, **kwargs) 
        
    @torch.no_grad()
    def get_input(self, batch, key, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        # implementation right from DDPM
        x = batch[key]  # 通过 self.first_stage_key == 'image' 取出 image
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')  #  x: (B,3,256,256)
        x = x.to(memory_format=torch.contiguous_format).float()

        # override from LatentDiffusion
        if bs is not None:
            x = x[:bs]  # 会和原来不一样??
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)  #  (B,3,64,64)
        # 调用VQModelInterface的encode方法 自带@torch.no_grad()
        z = self.get_first_stage_encoding(encoder_posterior).detach()  # z=posterior: (B,3,64,64)
        # 取出一个分离了计算图的副本，无法对 first_stage_model 求梯度，与@torch.no_grad()双重保障

        if self.model.conditioning_key is not None:  # = self.conditioning_key = 'crossattn'
            if cond_key is None:
                cond_key = self.cond_stage_key  # = 'fmri'
            if cond_key != self.first_stage_key:  # self.first_stage_key = 'image'
                if cond_key in ['caption', 'coordinates_bbox','fmri']:  # TODO: 可利用coordinates_bbox
                    xc = batch[cond_key]  # HERE!!!!!!!!!!!!!!取出 fmri数据  c:(B,4656)
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    raise NotImplementedError
                    # xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:  # not True or False --> False
                # cond 不能 train （即强制cond 只 encode） 才走这条支路；在get_input中直接算出fmri latent
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else: # xc 为 fmri 的 tensor batch
                    c = self.get_learned_conditioning(xc.to(self.device))  
                    # self.device从模型外传入，在运算时再决定哪些tensor需要放入GPU；一般使用GPU加速模型运算，其余语句可以不管
            else:
                c = xc  # HERE!!!!!!!!!!!!!!fmri data  c:(B,4656)
            if bs is not None:
                c = c[:bs]
            
            if self.use_positional_encodings:  # 默认为 False；这里positional_encodings是另外一回事？
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = self.__conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out  # z: (B,3,64,64)  c: (B,4654)

    def get_prompt_embedding(self, batch):  
 
        # _dataset = batch['dataset'][0] if batch['dataset'].dim() != 1 else batch['dataset']
        # _subject = batch['subject'][0] if batch['subject'].dim() != 1 else batch['subject']
        
        if self.enable_multi_subject and not self.enable_multi_dataset:
            # assert _subject <= 4 if _dataset == 0 else _subject <= 3 # 不同dataset的subject数量不同
            subject_idx = batch['subject']  # subject_idx 和 dataset_idx 应为实数,已在dotaloader中转换为Tensor
            prompt_embedding = self.embed_subject(subject_idx)
        elif self.enable_multi_subject and self.enable_multi_dataset:
            # (1) 一个nn.Embedding为不同的dataset编码; (2) 一个nn.Embedding为所有dataset的subject放在一起顺序编码
            dataset_idx = batch['dataset']
            subject_idx = batch['subject']
            dataset_embedding = self.embed_dataset(dataset_idx)  # (batch_size, dim_dataset_embedding)
            subject_embedding = self.embed_subject(subject_idx)  # (batch_size, dim_subject_embedding)
            prompt_embedding = torch.cat([dataset_embedding, subject_embedding])  # TODO: 注意维度
        else: # 本model只支持加入prompt的情况，即使用“multi-dataset” 或 “multi_subject + multi_dataset”
            raise NotImplementedError
        return prompt_embedding

    def shared_step(self, batch, **kwargs):  # Mind-Vis去掉了validation_step的调用

        z, c = self.get_input(batch, self.first_stage_key)  #  z:(B,3,64,64), c:(B,4656)
        prompt = self.get_prompt_embedding(batch)  # my implementation

        if self.return_cond:
            loss, cc = self(z, prompt, c)  # ref forword()
            return loss, cc
        else:    
            loss = self(z, prompt, c)  # ref forword()
            return loss

    def training_step(self, batch, batch_idx):  # 仅在DDPM中定义，LatentDiffusion中没有
        # self.train()  # 整个LatentDiffusion模型（作为子类），包括cond_stage_model
        # self.cond_stage_model.train()
        # CompVis 源代码里没有.train()的语句
            
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, 
                    logger=True, on_step=False, on_epoch=True)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx): 
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            z, _ = self.get_input(batch, self.first_stage_key)
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")


    # def save_images(self, all_samples, suffix=0):
    #     if self.output_path is not None:
    #         os.makedirs(os.path.join(self.output_path, 'val', f'{self.validation_count}_{suffix}'), exist_ok=True)
    #         for sp_idx, imgs in enumerate(all_samples):
    #             for copy_idx, img in enumerate(imgs[1:]):
    #                 img = rearrange(img, 'c h w -> h w c')
    #                 Image.fromarray(img).save(os.path.join(self.output_path, 'val', 
    #                                 f'{self.validation_count}_{suffix}', f'test{sp_idx}-{copy_idx}.png'))
                                    
    # def full_validation(self, batch, random_state=None):
    #     print('###### run full validation! ######\n')
    #     grid, all_samples, random_state = self.generate(batch, ddim_steps=self.ddim_steps, num_samples=5, limit=None, state=random_state)
    #     metric, metric_list = self.get_eval_metric(all_samples)
    #     # self.save_images(all_samples, suffix='%.4f'%metric[-1])
    #     metric_dict = {f'val/{k}_full':v for k, v in zip(metric_list, metric)}
    #     self.logger.log_metrics(metric_dict, step=self.trainer.current_epoch)  # FIXME: 记录steps的罪魁祸首！
    #     grid_imgs = Image.fromarray(grid.astype(np.uint8))
    #     self.logger.log_image(key=f'samples_test_full', images=[grid_imgs], step=self.trainer.current_epoch)
    #     if metric[-1] > self.best_val:
    #         self.best_val = metric[-1]
    #         torch.save(
    #             {
    #                 'model_state_dict': self.state_dict(),
    #                 # 'config': self.main_config,
    #                 'state': random_state

    #             },
    #             os.path.join(self.output_path, 'checkpoint_best.pth')
    #         )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # if self.trainer.current_epoch == 0:
        #     return
        assert batch_idx == 0, '\U0001F630'*10
        if self.global_rank == 0:
            start_time = time.time()
            print('\U0001F3A8'*2,f'Epoch{self.trainer.current_epoch}: generating images and calculating metrics',end='...') 
        n_partial_sampling = None if (self.trainer.current_epoch+1)%100==0 else 5  # 每100个epoch对rank0的所有sample进行记录
        # n_random_sampling = 5 if (self.trainer.current_epoch+1)%50==0 else 3
        n_random_sampling = 3  # 采样3个够了
        ##############################################
        # if self.global_rank == 0:
        #     start_time_generating = time.time()
        #     print('\U0001F559 generating images...')
        ##############################################
        grid, all_samples, state = self.generate(batch, batch_idx, ddim_steps=self.ddim_steps, 
                                                 num_samples=n_random_sampling, limit=n_partial_sampling)
        #  all_samples:(B,num_samples+1,3,256,256)
        ##############################################
        # if self.global_rank == 0:
        #     end_time_generating = time.time()
        #     print(f"\U0001F559 generating images: costs {(end_time_generating - start_time_generating)/60:.2f} mins")
        #     print('\U0001F559 getting eval metrics...')
        ##############################################
        all_samples = all_samples.cpu().numpy()
        metrics, metric_keys = self.get_eval_metric(all_samples, avg=self.eval_avg) # 耗时
        # ['mse', 'pcc', 'ssim', 'psm', 'top-1-class', 'top-1-class(max)'] 元素均为标量
        # metric_dict = {f'val/{k}':v for k, v in zip(metric_keys, metrics)}
        ##############################################
        # if self.global_rank == 0:
        #     end_time_getting_metrics = time.time()
        #     print(f"\U0001F559 getting eval metrics: costs {(end_time_getting_metrics - end_time_generating)/60:.2f} mins")
        #     print('\U0001F559 logging images...')
        ##############################################
        metric_dict = {k:v for k, v in zip(metric_keys, metrics)}
        metric_dict_log = {f'val/{k}':v for k,v in metric_dict.items()}
        # self.logger.log_metrics(metric_dict, step=self.trainer.current_epoch)  
        self.log_dict(metric_dict_log, 
                    logger=True, on_step=False, on_epoch=True)  # rank 0 only!!!会在当前epoch平均
        if self.global_rank == 0 and batch_idx == 0:  # 只在rank0的batch0记录图像，只看一部分
            grid_image = Image.fromarray(grid.astype(np.uint8))
            # print(f'RANK{self.global_rank}-Batch{batch_idx}: logging images to the logger...')
            self.logger.log_image(key=f'samples_val', images=[grid_image])
            # print(f'RANK{self.global_rank}-Batch{batch_idx}: logging finished!')
            ##############################################
            # end_time_logging_images = time.time()
            # print(f"\U0001F559 logging images: costs {(end_time_logging_images - end_time_getting_metrics)/60:.2f} mins")
            ##############################################
        # naive_label = batch['naive_label']  # 150~199一对一的50个标量
        # subject_idx = batch['subject_idx']
        # return {'sampled_images':all_samples,  # (B,num_samples+1,3,256,256)
        #         'metric_dict':metric_dict,  # (6)
        #         'subject':subject_idx, # (B,)
        #         'naive_label':naive_label  # (B,)
        #         }
        if self.global_rank == 0:
            end_time = time.time()
            print(f'finished, costig {(end_time - start_time)/60:.1f} mins', '\U0001F3A8'*2)

    

    # def validation_epoch_end(self, validation_step_outputs):
    #     sampled_images = validation_step_outputs['sampled_images']
    #     metric_dict = validation_step_outputs['metric_dict']
    #     subject_idx = validation_step_outputs['subject']  # GOD数据集中5个subject按0~4索引
    #     naive_label = validation_step_outputs['naive_label']  # GOD数据集中50个类别的naive_label取值范围为150~199
    #     # 第0个epoch需要跑一下试试
    #     if (self.trainer.current_epoch % 50 == 0) or \
    #                     (metric_dict['top-1-class(max)'] > self.run_full_validation_threshold): # 0.25
    #         concated_images = self.concat_images(sampled_images, subject_idx, naive_label)  # (5,50,n+1,3,256,256)
    #         # 分subject存
    #         grid_images = []
    #         for i in concated_images.shape[0]:  # 5个subject
    #             images = rearrange(concated_images[i], 'b n c h w -> (b n) c h w')
    #             grid = make_grid(images, nrow=concated_images.shape[1])
    #             grid_rgb = rearrange(grid, 'c h w -> h w c')
    #             grid_PIL= Image.fromarray(grid_rgb.astype(np.uint8))
    #         grid_images.append(grid_PIL)
    #         # self.save_images()
    #         if self.trainer.current_epoch % 50 == 0:
    #             self.logger.log_image(key=f'val/sampled images',images=grid_images,
    #                                   step=self.trainer.current_epoch)
    #         else:  # metric_dict['top-1-class(max)'] > self.run_full_validation_threshold
    #             self.logger.log_image(key=f'val/sampled images: top-1-class(max) > {self.run_full_validation_threshold})',
    #                                   images=grid_images,step=self.trainer.current_epoch)
    #     else:
    #         return

    
    @torch.no_grad()
    def generate(self, batch, batch_idx, num_samples, ddim_steps=250, HW=None, limit=None, state=None):
        # fmri_embedding: n, seq_len, embed_dim
        
        if HW is None:
            shape = (self.channels, 
                self.image_size, self.image_size)  # (3,64,64)
        else:
            num_resolutions = len(self.ch_mult)
            shape = (self.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self
        sampler = PD_PLMSSampler(model)
        # sampler = DDIMSampler(model)
        
        if torch.cuda.is_available():
            state = torch.cuda.get_rng_state() if state is None else state
            torch.cuda.set_rng_state(state)
        else:
            state = torch.get_rng_state() if state is None else state
            torch.set_rng_state(state)

        # rng = torch.Generator(device=self.device).manual_seed(2022).set_state(state)
        # state = torch.cuda.get_rng_state() 
        model.eval()   
        all_samples = []
        with model.ema_scope():
            for count, item in tqdm(enumerate(zip(batch['fmri'], batch['image'], batch['subject'])),
                                    desc=f'Generating images...Rank{self.global_rank}-Batch{batch_idx}',
                                    total=limit):
                # print('\U0001F92C'*40)  
                # print('item0-->fmri.shape',item[0].shape)
                # print('item1-->image.shape',item[1].shape)
                # print('item2-->subject.shape',item[2].shape)
                if limit is not None:
                    if count >= limit:  # 10个样本
                        break
                latent = item[0] # fmri embedding
                gt_image = rearrange(item[1], 'h w c -> 1 c h w') # h w c
                prompt = self.embed_subject(item[2])  # p added!!!
                # print(f"RANK{self.global_rank}-Batch{batch_idx}-Sample{count}: rendering {num_samples} examples in {ddim_steps} steps.")
                latent = repeat(latent, 'c d -> n c d', n=num_samples).to(self.device)
                p = repeat(prompt, 'd -> n d', n=num_samples).to(self.device)
                c = model.conditioning_forward(latent,p)
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                prompt=p,  # p added!!!
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0,min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image.detach(), x_samples_ddim.detach()], dim=0)) # put groundtruth at first
        all_samples = (255*(torch.stack(all_samples))).to(torch.uint8)
        # display as grid
        # grid = torch.stack(all_samples, 0)
        grid = rearrange(all_samples, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # model = model.to('cpu')
        return grid, all_samples, state

    @torch.no_grad()  # TODO
    def generate_paralle(self, batch, batch_idx, ddim_steps=250, HW=None, state=None,
                         num_samples=4, n_partial_batch: Union[int,None]=6,  n_paralle=3):
        if n_partial_batch is not None:
            assert n_partial_batch % n_paralle == 0, \
                f'partial_batch ({n_partial_batch}) should be divided by n_paralle ({n_paralle})'
            n_group = n_partial_batch / n_paralle
            partial_fmri = batch['fmri'][:n_partial_batch]
            partial_img = batch['image'][:n_partial_batch]  # (n_partial_batch, C, H, W)
            partial_sub = batch['subject'][:n_partial_batch]
        else: 
            partial_fmri = batch['fmri']    
            partial_img = batch['image']  # (B, C, H, W)
            partial_sub = batch['subject']
            n_group = partial_sub.shape[0] / n_paralle  # FIXME

        assert num_samples * n_paralle <= 12, \
                f'num_samples({num_samples}) * n_paralle({n_paralle}) cannot exceed 12'
        
        if HW is None:
            shape = (self.channels, self.image_size, self.image_size)  # (3,64,64)
        else:
            num_resolutions = len(self.ch_mult)
            shape = (self.channels,
                    HW[0] // 2 ** (num_resolutions - 1), HW[1] // 2 ** (num_resolutions - 1))

        if torch.cuda.is_available():
            state = torch.cuda.get_rng_state() if state is None else state
            torch.cuda.set_rng_state(state)
        else:
            state = torch.get_rng_state() if state is None else state
            torch.set_rng_state(state)
        
        model = self
        sampler = PD_PLMSSampler(model)
        model.eval()
        all_samples = []
        grouped_fmri = partial_fmri.reshape()  # TODO

        n=1
        with model.ema_scope():
            grouped_fmri = [batch['fmri'][i:i + n] for i in range(0, batch['fmri'].size(0), n)]
            grouped_images = [batch['image'][i:i + n] for i in range(0, batch['image'].size(0), n)]
            grouped_subjects = [batch['subject'][i:i + n] for i in range(0, batch['subject'].size(0), n)]

            for group_idx, (fmri_group, image_group, subject_group) in enumerate(zip(grouped_fmri, grouped_images, grouped_subjects)):
                latent_group = torch.stack(fmri_group)  # (g, [xx,)
                gt_image_group = rearrange(torch.stack(image_group), 'g h w c -> g c h w')
                prompt_group = torch.stack([self.embed_subject(subj) for subj in subject_group])  # (g, [xx,)

                sub_batch_size = n
                # print(f"RANK{self.global_rank}-Batch{batch_idx}-Group{group_idx}: rendering {sub_batch_size} examples in {ddim_steps} steps.")
                latent_group = latent_group.to(self.device)
                prompt_group = prompt_group.to(self.device)

                c = model.conditioning_forward(latent_group, prompt_group)
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                prompt=prompt_group,
                                                conditioning=c,
                                                batch_size=sub_batch_size,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                gt_image_group = torch.clamp((gt_image_group + 1.0) / 2.0, min=0.0, max=1.0)

                all_samples.extend([torch.cat([gt.detach().cpu(), x.detach().cpu()], dim=0) for gt, x in zip(gt_image_group, x_samples_ddim)])

        # Display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=sub_batch_size + 1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8), state

    
    @torch.no_grad() 
    def test_step(self, batch, batch_idx):
        # if self.subset == 'train' and batch_idx != 0:
        #     return
        start_time_generating = time.time()
        print('\U0001F3A8'*2,f'Epoch{self.global_step}: generating images',end='......') 
        n_partial_sampling = 2 if self.subset == 'train' else None # 默认的batch_size为50
        num_samples = 5  
        # ##############################################################
        # all_samples = rearrange(torch.clamp((batch['image']+1.0)/2.0,min=0.0,max=1.0), 'b h w c -> b c h w')
        # all_samples = repeat(all_samples, 'b c h w -> b n c h w', n=num_samples+1)
        # all_samples = (255. * all_samples).to(torch.uint8)
        # ##############################################################
        grid, all_samples, state = self.generate(batch, batch_idx, ddim_steps=self.ddim_steps, 
                                                 num_samples=num_samples, limit=n_partial_sampling)
        assert all_samples.dtype == torch.uint8, f'all_samples.dtype={all_samples.dtype}' # FIXME
        # all_samples: (B, 1+n_random_samples, 3, 256, 256), [0,255], (torch.uint8)
        end_time_generating = time.time()
        print(f"costs {(end_time_generating - start_time_generating)/60:.2f} mins")
        print('\U0001F4C8'*2, 'calculating eval metrics (fid, ssim, top-5-acc and top-50-acc)',end='......')
        
        all_imgs = all_samples/255.  # [0,1], torch.float32
        assert all_samples.dtype == torch.uint8, f'all_samples.dtype={all_samples.dtype}' # FIXME
        B, N, C, H, W = all_imgs.shape
        assert num_samples == N - 1
        # target = torch.tensor(sample_imgs[:,0,:,:,:], device=torch.device(self.device))
        target_imgs = all_imgs[:,0,:,:,:]  # (B,C,H,W) (torch.uint8) [0,255]
        sampled_imgs = all_imgs[:,1:,:,:,:]
        
        # fid 对两个set进行比较,不用等长?
        self.fid.update(target_imgs, real=True)  # (torch.uint8) [0,255] 
        self.fid.update(sampled_imgs.reshape(B*num_samples,C,H,W), real=False)
        fid = self.fid.compute()
        # 其余指标必须返回成对(pred,target)
        
        ssim_list = []
        top_5_acc_list = []
        top_50_acc_list = []
        # def topk_acc(output, target, topk=(1,)):
        #     maxk = max(topk)
        #     batch_size = target.size(0)

        #     _, pred = output.topk(maxk, 1, True, True)
        #     pred = pred.t()
        #     correct = pred.eq(target.view(1, -1).expand_as(pred))

        #     res = []
        #     for k in topk:
        #         correct_k = correct[:k].reshape(-1).float().sum(0)
        #         res.append(correct_k.mul_(1.0 / batch_size))
        #     return res
        # from sklearn.metrics import top_k_accuracy_score

        label = batch['imagenet_1k_class'][:n_partial_sampling] if n_partial_sampling is not None else batch['imagenet_1k_class']
        for i in range(num_samples):
            ssim = self.ssim(sampled_imgs[:,i,:,:,:], target_imgs)  # scalar
            ssim_list.append(ssim.cpu().numpy())
            pred_logits = self.classifier(self.classifier_img_transform(sampled_imgs[:,i,:,:,:]))  # (B,1000)
            top_5_acc = self.top_5_acc(pred_logits, label)
            # top_5_acc_wtf = topk_acc(pred_logits,label,topk=(5,))
            top_5_acc_list.append(top_5_acc.cpu().numpy())
            top_50_acc = self.top_50_acc(pred_logits, label) 
            # top_50_acc_wtf = topk_acc(pred_logits,label,topk=(50,))
            top_50_acc_list.append(top_50_acc.cpu().numpy())
        end_time_getting_metrics = time.time()
        print(f"costs {(end_time_getting_metrics - end_time_generating)/60:.2f} mins")
        print('\U0001F4E4'*2, 'logging metrics', end='......')
        ssim_avg = np.mean(ssim_list)
        top_5_acc_avg = np.mean(top_5_acc_list)
        top_50_acc_avg = np.mean(top_50_acc_list)
        metric_dict = {'fid': fid,
                       'ssim': ssim_avg,
                       'top-5-acc':top_5_acc_avg,
                       'top-50-acc':top_50_acc_avg}
        metric_dict_log = {f'{self.subset}/{k}':v for k,v in metric_dict.items()}
        self.log_dict(metric_dict_log, 
                    logger=True, on_step=False, on_epoch=True)
        end_time_logging = time.time()
        print(f"costs {(end_time_logging - end_time_getting_metrics)/60:.2f} mins")
        if n_partial_sampling is not None:
            return {'sampled_images': all_samples[:n_partial_sampling],
                    'subject': batch['subject'][:n_partial_sampling],
                    'naive_label': batch['naive_label'][:n_partial_sampling]}
        return {'sampled_images': all_samples,
                'subject': batch['subject'],
                'naive_label': batch['naive_label']}

    def on_test_start(self):
        # pass
        assert self.subset is not None, \
            '\U0001F605'*2 + 'My fault! but add this attribute in your script plz!' + '\U0001F62C'*2
        print('\n','\U0001F924'*2, f' {self.subset} set | on test start: initializing metrics......')
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(self.device)
        self.top_5_acc = MulticlassAccuracy(num_classes=1000,top_k=5).to(self.device)
        # self.top_5_acc_train = MulticlassAccuracy(num_classes=200,top_k=5)
        # self.top_5_acc_valid = MulticlassAccuracy(num_classes=50,top_k=5)
        self.top_50_acc = MulticlassAccuracy(num_classes=1000,top_k=50).to(self.device)
        self.classifier = vit_l_16(weights=ViT_L_16_Weights.DEFAULT).to(self.device)
        self.classifier_img_transform = ViT_L_16_Weights.DEFAULT.transforms().to(self.device)

    def test_epoch_end(self, test_step_outputs):
        start_time = time.time() 
        print('\U0001F4E4'*2, 'logging images', end='......')
        sampled_images_list = []
        subject_idx_list = []
        naive_label_list = []
        for step_output in test_step_outputs:
            sampled_images_list.append(step_output['sampled_images'])
            subject_idx_list.append(step_output['subject'])
            naive_label_list.append(step_output['naive_label'])
        sampled_images = torch.cat(sampled_images_list)
        assert sampled_images.dtype == torch.uint8, f'sampled_images.dtype={sampled_images.dtype}' # FIXME
        subject_idx = torch.cat(subject_idx_list)  # GOD数据集中5个subject按0~4索引
        naive_label = torch.cat(naive_label_list)  # GOD数据集中50个类别的naive_label取值范围为150~199

        grouped_images = self.grouping_images(sampled_images, subject_idx, naive_label, n_partial_sampling=10)
        assert grouped_images.dtype == torch.uint8, f'grouped_images.dtype={grouped_images.dtype}' # FIXME
        # (5,50,n+1,3,256,256) [0,255] torch.uint8
        # 分subject存
        grid_images = []
        for i in range(grouped_images.shape[0]):  # 5个subject分别make grid
            images = rearrange(grouped_images[i], 'b n c h w -> (b n) c h w')
            grid = make_grid(images, nrow=grouped_images.shape[2])
            grid_rgb = rearrange(grid, 'c h w -> h w c')
            assert grid_rgb.dtype == torch.uint8, f'grid_rgb.dtype={grid_rgb.dtype}' # FIXME
            grid_PIL= Image.fromarray(grid_rgb.cpu().numpy())
            grid_images.append(grid_PIL)

        self.logger.log_image(key=f'{self.subset}/sampled images', 
                              images=grid_images, 
                              caption=['sub1','sub2','sub3','sub4','sub5'])
        end_time = time.time()
        print(f"costs {(end_time - start_time)/60:.2f} mins")

    def grouping_images(self, sampled_images, subject_idx, naive_label, n_partial_sampling=10):
        '''
        暂时适用于GOD数据集中每个subject的每张gt图只有一个sample的情况
        '''
        # sampled_images: (250+,1+num_samples,3,256,256)
        # subject_idx: (250+,)
        # naive_label: (250+,)
        # 按照subject_idx取samples
        _, counts = torch.unique(subject_idx, return_counts=True)
        num_subjects = len(counts)
        print('\U0001F64B'*2, f' number of subjects is: {num_subjects}')
        grouped_image_list = []
        for i in range(num_subjects):
            selected_images = sampled_images[subject_idx == i]
            if self.subset=='train':  # 取出一部分看看就行
                grouped_image_list.append(selected_images[:n_partial_sampling]) 
            else:
                grouped_image_list.append(selected_images)
            # selected_naive_labels = naive_label[subject_idx == i]

            # sorted_naive_labels, indices= torch.sort(selected_naive_labels)
            # sorted_images = selected_images[indices]
            # sorted_images = torch.gather(selected_images,0,indices)

            # unique_sorted_naive_labels, unique_indices = np.unique(sorted_naive_labels, return_index=True)
            # xx = sorted_naive_labels[unique_indices] # 应该与unique_sorted_naive_label一致
            # unique_sorted_images = sorted_images[unique_indices]
            # grouped_image_list.append(unique_sorted_images)
        grouped_images = torch.stack(grouped_image_list)  # (5,50,3,n+1,256,256)

        return grouped_images

        

    
###########################################################################
###########################################################################

from dc_ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
from dc_ldm.modules.diffusionmodules.openaimodel import (UNetModel, 
                                                         Upsample,
                                                         Downsample,
                                                         AttentionBlock, 
                                                         TimestepBlock, 
                                                         ResBlock, 
                                                         TimestepEmbedSequential)
from dc_ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
)


class PDBasicTransformerBlock(BasicTransformerBlock): # 3 norm in!
    def __init__(self, dim, n_heads, d_head, dropout=0., prompt_dim=16, 
                 context_dim=None, gated_ff=True, checkpoint=True, cond_scale=1.):
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint, cond_scale)
      
        self.norm1 = PDLayerNorm(dim,prompt_dim)
        self.norm2 = PDLayerNorm(dim,prompt_dim)
        self.norm3 = PDLayerNorm(dim,prompt_dim)

    def forward(self, x, p, context=None):
        used_params = [p for p in self.parameters() if p.requires_grad]
        return checkpoint(self._forward, (x, p, context), used_params, self.checkpoint)

    def _forward(self, x, p, context=None):
        x = self.attn1(self.norm1(x, p)) + x
        x = self.attn2(self.norm2(x, p), context=context) + x
        x = self.ff(self.norm3(x, p)) + x
        return x


class PDSpatialTransformer(SpatialTransformer): # 4 norm in!
    """
    T
    """
    def __init__(self, in_channels, n_heads, d_head, prompt_dim=16,
                 depth=1, dropout=0., context_dim=None, cond_scale=1.):
        super().__init__(in_channels, n_heads, d_head, 
                         depth, dropout, context_dim, cond_scale)

        self.norm = PDGroupNorm(32,in_channels,prompt_dim) # 1 norm here!

        inner_dim = n_heads * d_head
        self.transformer_blocks = nn.ModuleList(
            [PDBasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, prompt_dim=prompt_dim,
                                      context_dim=context_dim,cond_scale=cond_scale)
                for d in range(depth)]
        ) # 3 norms here!

    def forward(self, x, p, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x, p)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, p, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class PDResBlock(ResBlock): # 2 instances of PDGroupNorm ins
    """
    A 
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        prompt_dim = 16,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__(channels,emb_channels,dropout,
                         out_channels,use_conv,use_scale_shift_norm,dims,use_checkpoint,up,down)

        # 1/2 norm in!
        # self.in_layers = nn.Sequential(
        #     PDGroupNorm(32,channels,prompt_dim),
        #     nn.SiLU(),
        #     conv_nd(dims, channels, self.out_channels, 3, padding=1),
        # )
        self.in_layers = nn.ModuleDict({
            '0': PDGroupNorm(32,channels,prompt_dim),
            '1': nn.SiLU(),
            '2': conv_nd(dims, channels, self.out_channels, 3, padding=1)})
        # self.in_norm = PDGroupNorm(32,channels,prompt_dim)
        # self.silu = nn.SiLU()
        # self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        # 2/2 norm in!
        # self.out_layers = nn.Sequential(
        #     PDGroupNorm(32,self.out_channels,prompt_dim),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     zero_module(
        #         conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        #     ),
        # )
        self.out_layers = nn.ModuleDict({
            '0': PDGroupNorm(32,self.out_channels,prompt_dim),
            '1': nn.SiLU(),
            '2': nn.Dropout(p=dropout),
            '3': zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))})
        # self.out_norm = PDGroupNorm(32,self.out_channels,prompt_dim)
        # # self.silu
        # self.dp = nn.Dropout(p=dropout)
        # self.out_conv = zero_module(
        #         conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))

    def forward(self, x, p, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, p, emb), self.parameters(), self.use_checkpoint
        )  


    def _forward(self, x, p, emb):
        # print('\U0001F628'*40)
        # print('x.shape',x.shape)
        # print('p.shape',p.shape)
        # print('emb.shape',emb.shape)
        in_norm = self.in_layers['0']
        in_silu = self.in_layers['1']
        in_conv = self.in_layers['2']
        if self.updown:
            # in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            # h = in_rest(x,p)  # PDGroupNorm() and nn.SiLU(),
            # h = self.h_upd(h)
            # x = self.x_upd(x)
            # h = in_conv(h)
            #################
            h = in_silu(in_norm(x,p))
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            # print('\U0001F383'*40)  # FIXME
            # print('x.shape',x.shape)
            # print('p.shape',p.shape)
            # print('emb.shape',emb.shape)
            # print('type(self.in_layers).__name__:',type(self.in_layers).__name__)
            # import ipdb; ipdb.set_trace()
            # h = self.in_layers(x, p) #  x:(5,192,64,64) p:(5,16) emb(5,768)
            h = in_conv(in_silu(in_norm(x,p)))
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        out_norm = self.out_layers['0']
        out_silu = self.out_layers['1']
        out_dp = self.out_layers['2']
        out_conv = self.out_layers['3']
        if self.use_scale_shift_norm:  # False so omit！
            # out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h, p) * (1 + scale) + shift
            h = out_conv(out_dp(out_silu(h)))
        else:
            h = h + emb_out
            # h = self.out_layers(h, p)
            h = out_norm(h, p)
            h = out_conv(out_dp(out_silu(h)))
        return self.skip_connection(x) + h



# TODO: 2 or 6 norm in!!
class PDTimestepEmbedSequential(TimestepEmbedSequential):
    """
    如不设构造函数, 则同父类的特性(调用父类的构造函数), 即调用父类的第一个父类的构造函数
    for containers, 不用在构造函数中传入新加入参数prompt_dim
    """

    def forward(self, x, p, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):  # 2 norm in ResBlock (as the sub class)!
                x = layer(x, p, emb)  
            elif isinstance(layer, PDSpatialTransformer):  # 4 norm in!
                x = layer(x, p, context)
            else:
                # raise NotImplementedError
                x = layer(x)  # AttentionBlock, conv_nd(), Downsample
        return x


class PDUNetModel(UNetModel):  # 109 norm instances in!
    """
    The full UNet model with attention and timestep embedding.
   
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        prompt_dim = 16,  # here I am!
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        cond_scale=1.0,
        global_pool=False,
        use_time_cond=False
    ):
        super().__init__(image_size,in_channels,model_channels,out_channels,num_res_blocks,attention_resolutions,
                         dropout,channel_mult,conv_resample,dims,num_classes,use_checkpoint,use_fp16,num_heads,
                         num_head_channels,num_heads_upsample,use_scale_shift_norm,resblock_updown,
                         use_new_attention_order,use_spatial_transformer,transformer_depth,context_dim,n_embed,
                         legacy,cond_scale,global_pool,use_time_cond)

        time_embed_dim = model_channels * 4
        # 40 norm in input_blocks
        self.input_blocks = nn.ModuleList(
            [
                PDTimestepEmbedSequential( 
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)  # dim=2
                    # (in_channels, model_channels, 3) 对应传入nn.Convxd(in_channels, out_channels, kernel_size)
                    # (3, 192, 3)
                )
            ]
        ) # 0 norm in List[0], yet appending...
        self._feature_size = model_channels  # 192
        input_block_chans = [model_channels]  # [192]
        ch = model_channels  # 192
        ds = 1
        for level, mult in enumerate(channel_mult): # (1,2,3,5)  mult是channel 增多的倍数
            for _ in range(num_res_blocks): # num_res_blocks=2 ，两条旁路
                layers = [
                    PDResBlock(  # 2 norm in! 
                        ch,  # 192
                        time_embed_dim,  # 192*4
                        dropout,  # 0
                        prompt_dim,
                        out_channels=mult * model_channels,  # 
                        dims=dims,  # 2
                        use_checkpoint=use_checkpoint,  # False
                        use_scale_shift_norm=use_scale_shift_norm,  # False
                    )  # 2 norm
                ]
                ch = mult * model_channels  # 随循环变化
                if ds in attention_resolutions:  # 1 in (8,4,2) --> Flase
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else PDSpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,cond_scale=cond_scale
                        )
                    )
                self.input_blocks.append(PDTimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    PDTimestepEmbedSequential(
                        PDResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        # 8 norms in!
        self.middle_block = PDTimestepEmbedSequential(
            PDResBlock(  # 2 norms in!
                ch,
                time_embed_dim,
                dropout,
                prompt_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else PDSpatialTransformer(  # 4 norms in!
                            ch, num_heads, dim_head, prompt_dim,
                            depth=transformer_depth, context_dim=context_dim, cond_scale=cond_scale
                        ),
            PDResBlock(  # 2 norms in!
                ch,
                time_embed_dim,
                dropout,
                prompt_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    PDResBlock(  # 2 norms in!
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        prompt_dim,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else PDSpatialTransformer(  # 4 norms in！
                            ch, num_heads, dim_head, prompt_dim,
                            depth=transformer_depth, context_dim=context_dim,cond_scale=cond_scale
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        PDResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            prompt_dim,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(PDTimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 1 insatance of GroupNorm in
        # self.out = nn.Sequential(
        #     PDGroupNorm(32, ch, prompt_dim),  
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        # )
        self.out = nn.ModuleDict({'0': PDGroupNorm(32, ch, prompt_dim),
                                  '1': nn.SiLU(),
                                  '2': zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1))})

    def forward(self, x, p, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param p: an [N x 16] Tensor of prompt
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        if self.use_time_cond: # add time conditioning
            c = self.time_embed_condtion(context)
            assert c.shape[1] == 1, f'found {c.shape}'
            emb = emb + torch.squeeze(c, dim=1)
        # 以下4个部分都要增加额外输入p
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, p, emb, context)  # FIXME🎃
            hs.append(h)
        h = self.middle_block(h, p, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, p, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            out = self.out['2'](self.out['1'](self.out['0'](h ,p)))
            return out.contiguous()


class PDDiffusionWrapper(DiffusionWrapper): # 109 Norm
    def __init__(self, unet_config, conditioning_key, prompt_dim=16, global_pool=False):
        unet_config['params']['global_pool'] = global_pool
        super().__init__(unet_config, conditioning_key)
        self.diffusion_model = PDUNetModel(prompt_dim = prompt_dim, **unet_config['params'])  # 109 Norm
        # overrided ! from: dc_ldm.modules.diffusionmodules.openaimodel.UNetModel
        
    def forward(self, x, prompt, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
                out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':   # self.model.conditioning_key = 'crossattn'
            cc = torch.cat(c_crossattn, 1)  # 当有多种condition时，在sequence维度上concat
            out = self.diffusion_model(x, prompt, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + [c_concat], dim=1)
            cc = torch.cat([c_crossattn], dim=1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
    

###########################################################################
###########################################################################

from timm.models.vision_transformer import Attention
from timm.models.layers import DropPath, Mlp
from sc_mbm.mae_for_fmri import MAEforFMRI


class PDCondStageModel(nn.Module):
    '''
    只用于stageC, 会load
    不需要在模型内部载入任何超参, 外部(LDM)会统一载入stageB的所有state
    '''
    def __init__(self, prompt_dim=16, embed_dim=512, global_pool=False, **kwargs):
        super().__init__()
        # prepare pretrained fmri mae 
        self.mae = PDMAEforFMRI(prompt_dim=prompt_dim, global_pool = global_pool)
        # 不从config载入 norm_layer
        self.fmri_seq_len = self.mae.num_patches  # 291  (291*169=4656)=num_voxels
        self.fmri_latent_dim = self.mae.embed_dim  # 1024
        if global_pool == False: # False,   load_checkpoint中，为True时，不载入LayerNorm的参数
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True), # kernel_size=1, 相当于1*1卷积
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, embed_dim, bias=True)
        self.global_pool = global_pool

    def encode(self, x, p):
        return self.forward(x, p)

    def forward(self, x, p):
        # n, c, w = x.shape
        latent_crossattn = self.mae.encode(x, p)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn) # 将seq_len作为channel去reduce？意义不明
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out


class PDBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=PDLayerNorm, prompt_dim=16):
        super().__init__()
        self.norm1 = norm_layer(dim, prompt_dim) # No.1 in 2
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, prompt_dim) # No.2 in 2
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, p):
        # x:(B,291,1024)  p:(B,16)
        x1 = x + self.drop_path(self.attn(self.norm1(x, p)))
        out = x1 + self.drop_path(self.mlp(self.norm2(x1, p)))
        return out  # (B,291,1024)


class PDMAEforFMRI(MAEforFMRI):  #  49 norm instances in CLASS fmri_encoder ! not including the decoder! 
    """ 
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_voxels=4656, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=1., norm_layer=PDLayerNorm, prompt_dim=16, global_pool=False,
                 focus_range=None, focus_rate=None, img_recon_weight=1.0, 
                 use_nature_img_loss=False):
        super().__init__(num_voxels, patch_size, embed_dim, in_chans,
                 depth, num_heads, decoder_embed_dim, 
                 decoder_depth, decoder_num_heads,
                 mlp_ratio, nn.LayerNorm, focus_range, focus_rate, img_recon_weight, 
                 use_nature_img_loss)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            PDBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, prompt_dim=prompt_dim)
            for i in range(depth)])  #  2*24 instances of norm here!!
        
        self.norm = norm_layer(embed_dim, prompt_dim)  #  1 instance here!
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # --------------------------------------------------------------------------
        self.num_patches = self.patch_embed.num_patches
        self.global_pool = global_pool

    def encode(self, c, p):
        # c:(B,4656)  p:(B,16)
        if c.dim() == 2:
            c = torch.unsqueeze(c, dim=0)  # (B,1,4656)
        # embed patches
        c = self.patch_embed(c)  # (B,291,1024) (B,seq_len,dim)

        # add pos embed w/o cls token
        c = c + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks: # 不改变维度
            c = blk(c, p)
        if self.global_pool:  # False
            c = c.mean(dim=1, keepdim=True)
        c = self.norm(c, p)  # (B,291,1024)

        return c  


class PD_PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda") and torch.cuda.is_available():
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               prompt = None,  # p added!!!
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates = self.plms_sampling(prompt, conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(self, prompt, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, generator=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device, generator=generator)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running PLMS Sampling with {total_steps} timesteps")

        # iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        # for i, step in enumerate(iterator):
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(img, prompt, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()  # p added!
    def p_sample_plms(self, x, p, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):  # NO p added!!! p 和 c 同时出现
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, p, t, c)  # p added!
            else:
                x_in = torch.cat([x] * 2)
                p_in = torch.cat([p] * 2)  # p added!
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, p_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)  # 模型估计的t时刻噪音
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t


###########################################################################
###########################################################################

# for debug
if __name__ == "__main__":
    # batch = torch.tensor({})
    # model = LatentDiffusion
    # out = model().training_step(batch)
    from omegaconf import OmegaConf

    config = OmegaConf.load('pretrains/ldm/label2img/config.yaml')
    pd_model = PDfLDM(**config.model.params)

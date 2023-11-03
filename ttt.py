import torch
import torch.nn as nn
from pdnorm_model import PDNorm
from torch import Tensor
from dc_ldm.modules.diffusionmodules.util import conv_nd



class PDGroupNorm(PDNorm, nn.GroupNorm):
    
    def __init__(self, num_groups: int, num_channels: int, prompt_dim: int=16, eps: float = 0.00001, 
                 affine: bool = False,  # False --> super.__init__中取消初始化self.weight和self.bias
                 device=None, dtype=None) -> None:
        nn.GroupNorm.__init__(self, num_groups, num_channels, eps, affine, device, dtype)
        
        self.mlp_scale = nn.Sequential(nn.Linear(prompt_dim, num_channels), nn.SiLU())
        self.mlp_shift = nn.Sequential(nn.Linear(prompt_dim, num_channels), nn.SiLU())
    
    def forward(self, x: Tensor, prompt:Tensor) -> Tensor:
        # 在C维，分组计算mean和var  
        B,C,H,W = x.shape
        x_i_standard = torch.empty(x.shape)
        for i in range(self.num_groups):
            x_i = x[:,int(i*(C/self.num_groups)):int((i+1)*(C/self.num_groups)),:,:] # (B,c/g,h,w)
            mean = torch.mean(x_i, dim=1, keepdim=True)  
            var = torch.var(x_i, dim=1, keepdim=True, unbiased=False)  # (b,1,h,w)
            x_i_standard[:,int(i*(C/self.num_groups)):int((i+1)*(C/self.num_groups)),:,:] = \
                        (x_i-mean)/torch.sqrt(var+self.eps)  # (B,c/g,h,w)
        #####################
        scale = self.mlp_scale(prompt).reshape((B,C,1,1))  
        shift = self.mlp_shift(prompt).reshape((B,C,1,1))  # (B,prompt_dim)-->(B,C)-->(B,C,1,1)
        out = x_i_standard * scale + shift
        # out = F.group_norm(
        #     x, self.num_groups, scale, shift, self.eps)        
        return out

class InLayersModule(nn.Module):
    '''
    fuck you! nn.sequential!
    '''
    def __init__(self) -> None:
        super().__init__()
        self.norm = PDGroupNorm(32,192,16)
        self.silu = nn.SiLU()
        self.conv = conv_nd(2, 192, 192, 3, padding=1)

    def forward(self, x, p):
        out = self._conv(self.silu(self.norm(x,p)))

        return out

# # in_layers = nn.Sequential(
# #             PDGroupNorm(32,192,16),
# #             nn.SiLU(),
# #             conv_nd(2, 192, 192, 3, padding=1),
# #         )
# in_layers = nn.ModuleDict({'0': PDGroupNorm(32,192,16),
#                     '1': nn.SiLU(),
#                     '2': conv_nd(2, 192, 192, 3, padding=1)})
# x = torch.ones(5,192,64,64)
# p = torch.ones(5, 16)
# # out0 = in_layers[0](x, p)
# # print(out0.shape)
# # out1 = in_layers[1](out0)
# # print(out1.shape)

# # print(in_layers[2])
# # out2 = in_layers[2](out1)
# # print(out2.shape)

# # out = in_layers(x, p)
# out = in_layers['0'](x, p)
# print(out.shape)
x = torch.randint(0,255,(3,256,256))
from pdnorm_dataset import *

from pdnorm_model import *

class PDBasicTransformerBlock(BasicTransformerBlock): # 3 norm in!
    def __init__(self, dim, n_heads, d_head, dropout=0., prompt_dim=16, 
                 context_dim=None, gated_ff=True, checkpoint=True, cond_scale=1.):
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint, cond_scale)
      
        self.norm1 = PDLayerNorm(dim,prompt_dim)
        self.norm2 = PDLayerNorm(dim,prompt_dim)
        self.norm3 = PDLayerNorm(dim,prompt_dim)

    def forward(self, x, p, context=None):
        return checkpoint(self._forward, (x, p, context), self.parameters(), self.checkpoint)

    def _forward(self, x, p, context=None):
        x = self.attn1(self.norm1(x, p)) + x
        x = self.attn2(self.norm2(x, p), context=context) + x
        x = self.ff(self.norm3(x, p)) + x
        return x


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

# out = image_transform(x,subset='train')
print(type(x))


from turtle import forward
from unittest.mock import patch
from xml.dom.expatbuilder import Skipper
import torch 
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from .fa_block import FABlock,DulFABlock

class PatchMerge(nn.Module):
  def __init__(self,resolution, dim, norm_layer=nn.LayerNorm) -> None:
    super().__init__()
    self.resolution = resolution
    self.dim = dim
    self.norm = norm_layer(dim*4)
    self.reduction = nn.Linear(dim*4, dim*2, bias=False)
  
  def forward(self,x):
    """
    Args:
      x: (B H*W C)
    Returns:
      x: (B H*W C)
    """
    H, W = self.resolution
    B, L, C = x.shape
    assert(L == H*W)
    assert(H%2==0 and W%2==0)

    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
    
    x = self.norm(x)
    x = self.reduction(x)

    return x

class PatchExpand(nn.Module):
  def __init__(self, resolution, dim, norm_layer=nn.LayerNorm) -> None:
    super().__init__()
    self.resolution = resolution
    self.dim = dim
    self.expand = nn.Linear(dim, 2*dim, bias=False)
    self.norm = norm_layer(dim//2)

  def forward(self, x):
    H, W =self.resolution
    x = self.expand(x)
    B, L, C = x.shape
    assert(L == H*W)

    x = x.view(B, H, W, C)
    x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C//4)
    x = x.view(B, -1, C//4)
    x = self.norm(x)

    return x

class TokenEmbed(nn.Module):
  def __init__(self, img_size:int=244, patch_size:int=32, hidden=None) -> None:
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.N = img_size//patch_size
    self.L = patch_size**2
    hidden = hidden or 2*(self.L)
    self.linear_project = nn.Sequential(
      nn.Linear(self.L + 2, hidden),
      nn.LayerNorm(hidden),
      nn.Linear(hidden, self.L)
    )
    self.merge = PatchMerge((patch_size,patch_size),self.N**2)
  def forward(self,x):
    """
    Args:
      x: B C H W
    Returns:
      x: B L C
    """
    p1=p2 = self.img_size//self.patch_size
    B,C,H,W = x.shape
    x = rearrange(x, "b c (p1 h) (p2 w) -> b (p1 p2) (h w c)", p1=p1,p2=p2)
    pos = np.indices([self.N, self.N]).transpose(1,2,0).reshape(1,-1,2)
    pos = torch.from_numpy(pos).broadcast_to(x.shape[0],x.shape[1],2)
    x = torch.cat([x,pos],dim=-1)
    
    x = self.linear_project(x)
    x = rearrange(x,"b c l -> b l c")
    x = self.merge(x)

    return x

class Encoder(nn.Module):
  def __init__(self, img_size=224,in_chans=1,patch_size=32, mlp_ratio=4,
               mixer_ratio=1,drop=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.token_embed = TokenEmbed(img_size, patch_size)
    
    dim = 2*((img_size//patch_size)**2)
    res = patch_size//2
    self.stage1 = DulFABlock(dim,res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    self.merge1 = PatchMerge((res,res),dim,norm_layer)

    dim = dim*2
    res = res//2
    self.stage2 = DulFABlock(dim,res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    self.merge2 = PatchMerge((res,res),dim,norm_layer)

    dim = dim*2
    res = res//2

    self.stage3 = DulFABlock(dim,res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    self.merge3 = PatchMerge((res,res),dim,norm_layer)

    dim = dim*2
    res = res//2
    self.stage4 = FABlock(dim,res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)

    self.final_dim = dim
    self.final_res = res

  def forward(self,x):
    skip_temp = []
    x = self.token_embed(x)
    x = self.stage1(x)
    skip_temp.append(x)
    x = self.merge1(x)

    x = self.stage2(x)
    skip_temp.append(x)
    x = self.merge2(x)

    x = self.stage3(x)
    skip_temp.append(x)
    x = self.merge3(x)
    
    x = self.stage4(x)
    skip_temp.append(x)

    return tuple(skip_temp)

class Decoder(nn.Module):
  def __init__(self,dim,res,out_chans,img_size,patch_size,mlp_ratio=4, 
               mixer_ratio=1,drop=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()

    self.stage1 = FABlock(dim,res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)

    self.stage2 = nn.Sequential(
      PatchExpand((res,res),dim,norm_layer),
      DulFABlock(dim//2,4*res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    )
    dim = dim//2
    res = res*2

    self.stage3 = nn.Sequential(
      PatchExpand((res,res),dim,norm_layer),
      DulFABlock(dim//2,4*res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    )
    dim = dim//2
    res = res*2

    self.stage4 = nn.Sequential(
      PatchExpand((res,res),dim,norm_layer),
      DulFABlock(dim//2,4*res*res,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    )
    dim = dim//2
    res = res*2

    self.rearrange = nn.Sequential(
      PatchExpand((res,res),dim,norm_layer),
      Rearrange("b (h w c) (p1 p2) -> b c (p1 h) (p2 w)", 
                c=out_chans,p1=img_size//patch_size,p2=img_size//patch_size,
                h=patch_size,w=patch_size)
    )

  def forward(self,skip_temp):
    x = self.stage1(skip_temp[-1])
    x = self.stage2(x) + skip_temp[-2]
    x = self.stage3(x) + skip_temp[-3]
    x = self.stage4(x) + skip_temp[-4]
    x = self.rearrange(x)
    return x

class FAUnet(nn.Module):
  def __init__(self, img_size=224,in_chans=1,patch_size=32, mlp_ratio=4,mixer_ratio=1,
               drop=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
    super().__init__()
    self.encoder = Encoder(img_size,in_chans,patch_size, mlp_ratio,mixer_ratio,
                          drop, act_layer, norm_layer)
    self.decoder = Decoder(self.encoder.final_dim,self.encoder.final_res,in_chans,
                           img_size,patch_size,mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)

  def forward(self,x):
    return self.decoder(self.encoder(x))
    
def main():
  print('testing FAUnet module')
  x = torch.rand((2,1,224,224))
  model = FAUnet()
  print(model)
  y = model(x)
  print('x shape: ',x.shape)
  print('y shape: ',y.shape)

if __name__ == 'main':
  main()
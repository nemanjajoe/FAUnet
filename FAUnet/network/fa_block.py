import torch
import torch.nn as nn
import torch.fft as fft

class LinearTransform(nn.Module):
  """Linear transformation layer

  Applies matrix multiplications over sequence and hidden dimensions
  
  Attributes:
  """
  def __init__(self, in_features, out_features=None,act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    self.fc = nn.Linear(in_features, out_features)
    self.act = act_layer()
    self.drop = nn.Dropout(drop)

  def forward(self,x):
    x = self.fc(x)
    x = self.act(x)
    x = self.drop(x)

    return x

class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None,act_layer=nn.GELU, drop=0.):
    super().__init__()
    hidden_features = hidden_features or in_features
    out_features = out_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features,out_features)
    self.drop = nn.Dropout(drop)

  def forward(self,x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)

    return x

class MlpMixer(nn.Module):
  def __init__(self,dim,seq_len,mixer_ratio=1) -> None:
    super().__init__()
    self.token_mixer = nn.Sequential(
      nn.Linear(dim,mixer_ratio*dim),
      nn.GELU(),
      nn.Linear(mixer_ratio*dim,dim)
    )
    self.channal_mixer = nn.Sequential(
      nn.Linear(seq_len,mixer_ratio*seq_len),
      nn.GELU(),
      nn.Linear(mixer_ratio*seq_len,seq_len)
    )
    
  
  def forward(self,q_,k_):
    x = self.channal_mixer(k_.transpose(-2,-1)).transpose(-2,-1)
    x = self.token_mixer(q_)*x
    return x

class FAttention(nn.Module):
  def __init__(self, d_model,seq_len,mixer_ratio=1,norm_laer=nn.LayerNorm):
    super().__init__()
    self.mlp_mixer = MlpMixer(dim=d_model,seq_len=seq_len,mixer_ratio=mixer_ratio)
    self.norm = norm_laer(d_model)

  def forward(self,q,k,v):
    """
    q, k, v shape: (B L C)
    """
    assert(q.shape == k.shape == v.shape)
    z_q = fft.fft(fft.fft(q,dim=2),dim=1)
    z_k = fft.fft(fft.fft(k,dim=2),dim=1)
    x = self.mlp_mixer(torch.real(z_q), torch.real(z_k))
    x = fft.fft(fft.fft(x,dim=2),dim=1)
    x = torch.real(x)
    x = self.norm(x)
    x += v
    return x

class FABlock(nn.Module):
  def __init__(self, dim,seq_len,mlp_ratio=4,mixer_ratio=1, drop=0.,
               act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.dim = dim
    self.mlp_ratio = mlp_ratio
    self.norm1 = norm_layer(dim)
    self.qkv = nn.Linear(dim, dim*3, bias=True)
    self.f_att = FAttention(dim,seq_len,mixer_ratio,norm_layer)
    self.norm2 = norm_layer(dim)
    self.mlp = Mlp(dim, dim*mlp_ratio, dim, act_layer, drop)

  def forward(self, x):
    """
    Args:
      x: (B L C)
    Returns:
      x: (B L C)
    """
    B,L,C = x.shape
    temp = self.norm1(x)
    q,k,v = self.qkv(temp).reshape(B, L, 3, C).permute(2,0,1,3)
    att = self.f_att(q,k,v)
    x = x + att
    x = x + self.mlp(self.norm2(x))

    return x

class DulFABlock(nn.Module):
  def __init__(self, dim,seq_len,mlp_ratio=4,mixer_ratio=1, drop=0.,
               act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
    super().__init__()
    self.fa1 = FABlock(dim,seq_len, mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
    self.fa2 = FABlock(dim,seq_len, mlp_ratio,mixer_ratio,drop,act_layer,norm_layer)
  
  def forward(self,x):
    # print("before:",x.shape)
    x = self.fa2(self.fa1(x))
    # print("after:",x.shape)
    return x
    


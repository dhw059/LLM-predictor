import numpy as np
import pandas as pd
import torch
from torch import nn

# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32
device = torch.device("cuda")

# %%
class Embedder(nn.Module):
    def __init__(self,
                 d_model,
                 compute_device=device):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device
        elem_dir = 'data/element_properties'
        # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


# %%
class FractionalEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
        ):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model) # [5000, 512]
        pe[:, 0::2] = torch.sin(x /torch.pow(
            50,2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(
            50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution

        x = torch.clamp(x, min=1/self.resolution) 
        # 返回一个新张量，将输入input张量的每个元素舍入到最近的整数。
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]
        return out

# %%
class Encoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 frac=False,
                 compute_device=device):
        super().__init__()
        self.d_model = d_model
        self.fractional = frac
        self.compute_device = compute_device
        
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False).to(self.compute_device)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True).to(self.compute_device)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.])).to(self.compute_device)
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.])).to(self.compute_device)
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.])).to(self.compute_device)

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler  
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler
        x_src = x + pe + ple
        return x_src

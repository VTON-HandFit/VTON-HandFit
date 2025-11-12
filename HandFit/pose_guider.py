from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class FeatureEmbedder_LN(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.linear(x)
        
class FeatureEmbedder_onelinear_ln(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.linear(x)
            


class Handprocessor(nn.Module):
    def __init__(self, fourier_freqs=8):
        super().__init__()
        self.embedders = nn.ModuleDict({
            'mano_bps': FeatureEmbedder_LN(1024,768,512),
            'mano_6d': FeatureEmbedder_onelinear_ln(96, 512),
            #  'mano_2d': FeatureEmbedder_LN(42,384,512),
            'handtype': FeatureEmbedder_onelinear_ln(1, 512),  # Assuming handtype is categorical with 2 categories
            # 'dino_embedding': FeatureEmbedder_onelinear_ln(1536, 512),
        })
        self.mano_2d_mlp =FeatureEmbedder_LN(42,384,768)
        self.dino_mlp = FeatureEmbedder_onelinear_ln(1536, 768)
        # self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 for sin&cos, 4 for bbox

        total_embedding_dim = 512 * 3 # + self.position_dim  # 5 feature embeddings + 1 position embedding
        self.linears = nn.Sequential(
            FeatureEmbedder_onelinear_ln(total_embedding_dim,768)
        )
        # Null features for padding
        self.null_features = nn.ParameterDict({
            'mano_bps': nn.Parameter(torch.zeros(512)),
            'mano_6d': nn.Parameter(torch.zeros(512)),
            'mano_2d': nn.Parameter(torch.zeros(768)),
            'handtype': nn.Parameter(torch.zeros(512)),
            'dino_embedding': nn.Parameter(torch.zeros(768)),
            # 'bbox': nn.Parameter(torch.zeros(self.position_dim))
        })

    def forward(self, mano_bps, mano_6d, mano_2d, handtype, bbox, dino_embedding, mask=None, mask_bbox=None, mask_dino=None, mask_2d=None,decoupling=True):
        embeddings = []
        feature_dict = {
            'mano_bps': mano_bps,
            'mano_6d': mano_6d,
            # 'mano_2d': mano_2d,
            'handtype': handtype,
            # 'dino_embedding': dino_embedding
        }

        for key, feature in feature_dict.items():
            feature_embedding = self.embedders[key](feature)
            null_feature = self.null_features[key].expand_as(feature_embedding)
            feature_embedding = feature_embedding * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)) * null_feature
            # print(key, feature_embedding.abs().mean())
            embeddings.append(feature_embedding)


        combined_3d = torch.cat(embeddings, dim=-1)

        output_3d = self.linears(combined_3d)
        dino_embedding_mlp = self.dino_mlp(dino_embedding)

        dino_embedding_mlp = dino_embedding_mlp * mask_dino.unsqueeze(-1) + (1 - mask_dino.unsqueeze(-1)) * (self.null_features['dino_embedding'].expand_as(dino_embedding_mlp))

        mano_2d_embedding = self.mano_2d_mlp(mano_2d)
        mano_2d_embedding = mano_2d_embedding * mask_2d.unsqueeze(-1) + (1 - mask_2d.unsqueeze(-1)) * (self.null_features['mano_2d'].expand_as(mano_2d_embedding))

        if decoupling == True:
            mano_output = torch.cat([output_3d, mano_2d_embedding ], dim=1)
            return mano_output, dino_embedding_mlp 
        
        output = torch.cat([output_3d, mano_2d_embedding ,dino_embedding_mlp], dim=1)
        return output

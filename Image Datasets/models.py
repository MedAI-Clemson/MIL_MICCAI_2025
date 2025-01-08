import torch
from torch import nn
import math
from utils import pad

class MILNet(nn.Module):
    def __init__(self, encoder, pooling_method='attn', num_heads=2, num_features=512, num_out=1, drop_rate=0, debug=False):
        super(MILNet, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()
        self.num_heads = num_heads
        self.num_features = num_features
        self.num_out = num_out
        self.drop_rate = drop_rate
        self.subspace_size = num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.pool = pooling_method
        if self.pool == 'attn':
            self.pool_func = self.attn_pool
            self.attn_vecs = nn.Parameter(torch.randn(self.num_heads, self.subspace_size))
        elif self.pool == 'max':
            self.pool_func = self.max_pool
        elif self.pool == 'avg':
            self.pool_func = self.avg_pool
        else:
            raise NotImplementedError(f"{self.pool} pooling method has not been implemented. Use one of 'attn', 'max', or 'avg'")
        
        self.fc = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.num_features, self.num_out)
        )
        self.debug = debug
        
    def forward(self, x, num_imgs):
        # hn_img: the hidden representations of each img
        hn_img = self.encoder(x)
        print("hn_img shape:", hn_img.shape) if self.debug else None
        # expect [L*N, h_dim (num_features)]
        
        # find the hidden representation of entire study and corresponding attn score
        h_study, attn = self.pool_func(hn_img, num_imgs)
        # expect [N, num_heads, subspace_size], [L, N, num_heads]
        
        # feed to fully-connected layer for decoding
        output = self.fc(h_study)
        print("output shape:", output.shape) if self.debug else None
        # expect [N, output size]
        
        return output, attn
    
    def attn_pool(self, hn_img, num_imgs):
        # Calculate attention logits
        hn_img = hn_img.view(-1, self.num_heads, self.subspace_size)
        print("hn shape:", hn_img.shape) if self.debug else None
        # expect [L*N, num_heads, subspace size]
        print("query vector shape:", self.attn_vecs.shape) if self.debug else None
        # expect [num_heads, subspace size]
        
        alpha = (hn_img * self.attn_vecs).sum(axis=-1) / self._scale
        print("alpha shape:", alpha.shape) if self.debug else None
        # expect [L*N, num_heads]
        
        # normalized attention
        alpha = pad(alpha, num_imgs)
        print("alpha_pad shape:", alpha.shape) if self.debug else None
        # expect [L, N]?
        for ix, n in enumerate(num_imgs):
            alpha[n:, ix]=-50
        attn = torch.softmax(alpha, axis=0)
        print("attn shape:", attn.shape) if self.debug else None
        # expect [L, N, num_heads]
        
        # pool within subspaces
        h_query_pad = pad(hn_img, num_imgs) # reshapes the study with the number of images
        print("h_query_pad shape:", h_query_pad.shape) if self.debug else None
        # expect [L, N, num_heads, subspace size]
        h_study = torch.sum(h_query_pad * attn[...,None], axis=0)
        print("h_study shape:", h_study.shape) if self.debug else None
        # expect [N, num_heads, subspace size]
        
        h_study_wide = h_study.view(-1, self.num_features)
        print("h_study_wide shape:", h_study_wide.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study_wide, attn
            
    def max_pool(self, hn_img, num_imgs):
        hn_pad = pad(hn_img, num_imgs)
        print("hn_pad shape:", hn_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take max
        h_study = hn_pad.max(0).values
        print("h_study shape:", h_study.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study, None
    
    def avg_pool(self, hn_img, num_imgs):
        hn_pad = pad(hn_img, num_imgs)
        print("hn_pad shape:", hn_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take avg
        h_study = hn_pad.mean(0)
        print("h_study shape:", h_study.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_study, None

       
import torch
import torch.nn as nn
from random import sample
from torch.nn.modules.linear import Linear
import torchvision.models as models
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# encoder 
class MLPEncoder(nn.Module):
    def __init__(self, num_genes=10000, num_hiddens=128, p_drop=0.5):
        super().__init__()
        self.linear1 = nn.Linear(num_genes, 1024, bias=True)
        self.norm1 = nn.LayerNorm(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)
        self.linear2 = nn.Linear(1024, num_hiddens, bias=True)
        self.norm2 = nn.LayerNorm(num_hiddens)
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        return x   
class MoCoV3(nn.Module):
    def __init__(self, 
                 base_encoder, 
                 num_genes=10000, 
                 dim=16, 
                 m=0.999, 
                 T=0.2):
        """
        dim: feature dimension (default: 128)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        """
        super(MoCoV3, self).__init__()        
        self.m = m
        self.T = T
        # encoder model
        self.encoder_q = base_encoder(num_genes=num_genes, num_hiddens=dim)
        self.momentum_encoder = base_encoder(num_genes=num_genes, num_hiddens=dim)
        # create predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_m in zip(self.encoder_q.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_q.data * (1. - self.m)
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.mm(q, k.t())
        N = logits.shape[0]  # batch size per GPU
        labels = range(N)
        labels = torch.LongTensor(labels).cuda()        
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    def forward(self, x1, x2=None, is_eval=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            contrastive_loss
        """     
        if is_eval:
            k = self.encoder_q(x1)
            k = nn.functional.normalize(k, dim=1)            
            return k        
        q1 = self.predictor(self.encoder_q(x1))
        q2 = self.predictor(self.encoder_q(x2))
        # compute key features       
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
                    
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


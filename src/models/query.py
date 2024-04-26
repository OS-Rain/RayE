import math
import torch
import torch.nn as nn

class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim, num_class, multi_label):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Qdim, Mdim)
        self.fc_out = nn.Sequential(nn.Linear(Mdim, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, Q, K, V, mask=None):
        Q, K = self.WQ(Q), self.WK(K)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
        return torch.matmul(Attn, V)

    def get_emb(self, Q, K, V, mask=None):
        Q, K = self.WQ(Q), self.WK(K)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
        return self.fc_out(torch.matmul(Attn, V))

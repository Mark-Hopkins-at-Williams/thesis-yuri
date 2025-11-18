import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e, f):
        # x: (batch, seq_len, embed_dim)
        Q = e
        K = f
        V = f

        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (e.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        #print(torch.round(weights * 100) / 100)
        output = torch.matmul(weights, V)

        return output, weights

# ---- Demo ----
if __name__ == '__main__':
    batch_size, seq_len, embed_dim = 2, 5, 16
    x = torch.randn(batch_size, seq_len, embed_dim)

    attn = SimpleAttention()
    out, weights = attn(x, x)

    print("Output shape:", out.shape)       # (2, 5, 16)
    print("Attention weights shape:", weights.shape)  # (2, 5, 5)

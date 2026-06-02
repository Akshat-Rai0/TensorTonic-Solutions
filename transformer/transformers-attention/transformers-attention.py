import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(K.shape[-1])

    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)
    return output
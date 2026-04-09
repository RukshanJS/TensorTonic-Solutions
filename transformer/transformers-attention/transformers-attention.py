import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    dot_product=torch.matmul(Q,K.transpose(-2,-1))
    dot_product/=math.sqrt(K.size(-1))
    dot_product=F.softmax(dot_product,dim=-1)
    output=torch.matmul(dot_product,V)
    return output
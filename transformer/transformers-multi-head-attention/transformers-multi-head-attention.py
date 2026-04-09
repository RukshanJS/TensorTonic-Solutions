import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    B,L,d_model=Q.shape
    h=num_heads
    d_k=d_model//h

    Q_proj=np.matmul(Q,W_q)
    K_proj=np.matmul(K,W_k)
    V_proj=np.matmul(V,W_v)

    Q_proj=Q_proj.reshape(B,L,h,d_k)
    Q_proj=Q_proj.transpose(0,2,1,3)

    K_proj=K_proj.reshape(B,L,h,d_k)
    K_proj=K_proj.transpose(0,2,1,3)

    V_proj=V_proj.reshape(B,L,h,d_k)
    V_proj=V_proj.transpose(0,2,1,3)

    K_t=K_proj.transpose(0,1,3,2)
    scores=np.matmul(Q_proj,K_t)/np.sqrt(d_k)

    attn=softmax(scores,axis=-1)
    out=np.matmul(attn,V_proj)
    out=out.transpose(0,2,1,3)
    out=out.reshape(B,L,d_model)
    out=np.matmul(out,W_o)
    return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Modelargs:
    dim:int = 4096
    n_layers: int=32
    n_heads: int=32 # No. of heads for the queries
    n_kv_heads: Optional[int] = None # No. of heads for K and V
    vocab_size: int=-1 # This will be set when we load tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None   #FeedForward Neural Network 
    norm_eps: float = 1e-5
    
    #Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device:str = None

def precompute_theta_pos_frequencies(head_dim:int,seq_len:int,device :str,theta:float=10000):
    ## As written in paper , Head_dim should be even
    assert head_dim%2 == 0
    
    ## Shape:(head_dim/2)
    theata_numerator = torch.arange(0,head_dim,2).float()
    
    ## Shape:(head_dim/2)
    theta = 1.0/(theta ** (theata_numerator/head_dim)).to(device)
    
    ## Construct the positions the m parameter
    m = torch.arange(seq_len, device=device)
    
    ## Multiply each theta by each position using the outer product
    #Shape (seq_len) outer_product*(head_dim/2) -> (seq_len,head_dim/2)
    freqs = torch.outer(m,theta).float()
    
    ## We can compute complex numbers in the polar form c= r * exp(i*m*theta) where r=1 as follows:
    
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor , freqs_complex:torch.Tensor ,device:str):
    ## (B, seq_len, H, Head_dim) -> (B, seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    ## (seq_len, Head_dim/2) -> (1, seq_len, 1, Head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    #(B, seq_len, H, Head_dim/2)*(1, seq_len, 1, Head_dim/2) = (B, seq_len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex
    
    # (B, seq_len, H, Head_dim/2) -> (B, seq_len, H, Head_dim/2, 2 )
    x_out = torch.view_as_real(x_rotated)

    # (B, seq_len, H, Head_dim/2, 2) -> (B, seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        ## Gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x:torch.Tensor):
        #(B, seq_len, Dim)
        ## rsqrt = 1/sqrt(x)
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor):
        #(Dim) * (B, seq_len, 1) = (B, seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

class Transformer(nn.module):
    
    def __init__(self,args:Modelargs)->None:
        
        super().__init__()
        
        assert args.vocab_size != -1
        
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim,eps=args.norm_eps)
        self.output = nn.Linear(args.dim,self.vocab_size,bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads,self.args.max_seq_len*2,device=self.args.device)
    
    def forward(self,tokens:torch.Tensor,start_pos:int):
        #(B,seq_len)
        
        batch_size,seq_len = tokens.shape
        assert seq_len == 1 # Only one token at a time can be processed
        
        #(B,seq_len) -> (B,seq_len,Dim)
        h = self.tok_embeddings(tokens)
        
        #Retrieve the pairs (m,theta) corresponding to the positions [start_pos,start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        
        #Consecutively apply all the encoder layers
        
        for layer in self.layers:
            h = layer(h, start_pos,freqs_complex)
            
        h = self.norm(h)
        output = self.output(h).float()
        
        return output
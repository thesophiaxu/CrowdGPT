import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        #att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1536
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

cfg = GPTConfig()
attnLayer = CausalSelfAttention(config=cfg)

c_attn_weights = attnLayer.c_attn.weight # 
c_attn_biases = attnLayer.c_attn.bias
c_proj_weights = attnLayer.c_proj.weight
c_proj_biases = attnLayer.c_proj.bias

torch.manual_seed(42)

c_attn_weights.data = torch.randn(3 * cfg.n_embd, cfg.n_embd, requires_grad=True)
c_attn_biases.data = torch.randn(3 * cfg.n_embd, requires_grad=True)
c_proj_weights.data = torch.randn(cfg.n_embd, cfg.n_embd, requires_grad=True)
c_proj_biases.data = torch.randn(cfg.n_embd, requires_grad=True)

input = torch.randn(1, 160, cfg.n_embd, requires_grad=True)

res = attnLayer.forward(input)
print(res)
print(res.size())

grad_Res = torch.randn_like(res)


import json
import struct
import os

with open("weights/test/c_attn_w.bin", 'wb') as file:
    values = c_attn_weights.detach().numpy()
    for single_value in values.T.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/c_attn_b.bin", 'wb') as file:
    values = c_attn_biases.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/c_proj_w.bin", 'wb') as file:
    values = c_proj_weights.detach().numpy()
    for single_value in values.T.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/c_proj_b.bin", 'wb') as file:
    values = c_proj_biases.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/inp.bin", 'wb') as file:
    values = input.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/grad_res.bin", 'wb') as file:
    values = grad_Res.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))

res.backward(grad_Res)

c_attn_weights_grads = attnLayer.c_attn.weight.grad
q, k, v = c_attn_weights_grads.split(cfg.n_embd, dim=0)
print(q.T)
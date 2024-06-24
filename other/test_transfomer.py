from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

# change seed
torch.manual_seed(42)

n_layer = 2
n_head = 4
n_embd = 128
n_ctx = 64
vocab_size = 65
dirname = 'weights/rand'

# initialize random model weights
weights_dict = dict()

weights_dict['transformer.wte.weight'] = torch.randn(vocab_size, n_embd)
weights_dict['transformer.wpe.weight'] = torch.randn(n_ctx, n_embd)

for layer in range(n_layer):
    weights_dict[f'transformer.h.{layer}.ln_1.weight'] = torch.ones(n_embd)
    weights_dict[f'transformer.h.{layer}.ln_1.bias'] = torch.zeros(n_embd)
    weights_dict[f'transformer.h.{layer}.attn.c_attn.weight'] = torch.randn(3 * n_embd, n_embd).T
    weights_dict[f'transformer.h.{layer}.attn.c_attn.bias'] = torch.randn(3 * n_embd)
    weights_dict[f'transformer.h.{layer}.attn.c_proj.weight'] = torch.randn(n_embd, n_embd).T
    weights_dict[f'transformer.h.{layer}.attn.c_proj.bias'] = torch.randn(n_embd)
    weights_dict[f'transformer.h.{layer}.ln_2.weight'] = torch.ones(n_embd)
    weights_dict[f'transformer.h.{layer}.ln_2.bias'] = torch.zeros(n_embd)
    weights_dict[f'transformer.h.{layer}.mlp.c_fc.weight'] = torch.randn(4 * n_embd, n_embd).T
    weights_dict[f'transformer.h.{layer}.mlp.c_fc.bias'] = torch.randn(4 * n_embd)
    weights_dict[f'transformer.h.{layer}.mlp.c_proj.weight'] = torch.randn(n_embd, 4 * n_embd).T
    weights_dict[f'transformer.h.{layer}.mlp.c_proj.bias'] = torch.randn(n_embd)

weights_dict['transformer.ln_f.weight'] = torch.randn(n_embd)
weights_dict['transformer.ln_f.bias'] = torch.randn(n_embd)
weights_dict['lm_head.weight'] = torch.randn(vocab_size, n_embd)

## initialize model
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
    block_size: int = 64
    vocab_size: int = 65 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

cfg = GPTConfig()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        self.lnOutput = self.ln_1(x)
        self.lnOutput.retain_grad()
        y = x + self.attn(self.lnOutput)
        z = y + self.mlp(self.ln_2(y))
        return z
    
# initialize weights
myBlock = Block(cfg)
myBlock.ln_1.weight.data = weights_dict['transformer.h.0.ln_1.weight']
myBlock.ln_2.weight.data = weights_dict['transformer.h.0.ln_2.weight']
myBlock.ln_1.bias.data = weights_dict['transformer.h.0.ln_1.bias']
myBlock.ln_2.bias.data = weights_dict['transformer.h.0.ln_2.bias']
myBlock.mlp.c_fc.weight.data = weights_dict['transformer.h.0.mlp.c_fc.weight'].T
myBlock.mlp.c_fc.bias.data = weights_dict['transformer.h.0.mlp.c_fc.bias']
myBlock.mlp.c_proj.weight.data = weights_dict['transformer.h.0.mlp.c_proj.weight'].T
myBlock.mlp.c_proj.bias.data = weights_dict['transformer.h.0.mlp.c_proj.bias']
myBlock.attn.c_attn.weight.data = weights_dict['transformer.h.0.attn.c_attn.weight'].T
myBlock.attn.c_attn.bias.data = weights_dict['transformer.h.0.attn.c_attn.bias']
myBlock.attn.c_proj.weight.data = weights_dict['transformer.h.0.attn.c_proj.weight'].T
myBlock.attn.c_proj.bias.data = weights_dict['transformer.h.0.attn.c_proj.bias']

myInput = torch.randn(1, 16, cfg.n_embd, requires_grad=True)
myOutput = myBlock(myInput)
dOutput = torch.randn_like(myOutput)
myOutput.backward(dOutput)


print('\nInput:')
print(myInput)
print(myInput.size())

print('\nOutput:')
print(myOutput)
print(myOutput.size())

print('\ndOutput:')
print(dOutput)
print(dOutput.size())

print('\ndInput:')
c_attn_weights_grads = myBlock.attn.c_attn.weight.grad
q, k, v = c_attn_weights_grads.split(cfg.n_embd, dim=0)
print(q.T)
print(q.T.size())

print('\ndInput:')
# print(myBlock.ln_1.weight.grad)
# print(myBlock.ln_1.weight.grad.size())
# print(myBlock.ln_1.bias.grad)
# print(myBlock.ln_1.bias.grad.size())
print(myInput.grad)
print(myInput.grad.size())

import struct

with open("weights/test/1L_input.bin", 'wb') as file:
    values = myInput.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/1L_dOutput.bin", 'wb') as file:
    values = dOutput.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))
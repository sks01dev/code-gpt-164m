import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ Single Causal Attention Head with KV Caching """
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer_past=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_kv = (k, v)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        if layer_past is None:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ v, present_kv

class MultiHeadAttention(nn.Module):
    """ Parallel Multi-Head Causal Attention """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer_past=None):
        head_outputs = []
        present_kvs = []
        for i, h in enumerate(self.heads):
            past_i = layer_past[i] if layer_past is not None else None
            out, present_kv = h(x, layer_past=past_i)
            head_outputs.append(out)
            present_kvs.append(present_kv)
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, present_kvs

class FeedForward(nn.Module):
    """ MLP with 4x Expansion and GELU Non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: LayerNorm -> Attention -> LayerNorm -> FeedForward """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, layer_past=None):
        sa_out, present_kv = self.sa(self.ln1(x), layer_past=layer_past)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, present_kv

class CodeGPT(nn.Module):
    """ Full Causal Transformer Language Model """
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, block_size=1024, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, layer_past=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        
        start_pos = 0
        if layer_past is not None:
            start_pos = layer_past[0][0][0].size(-2)
            
        pos_emb = self.position_embedding_table(torch.arange(start_pos, start_pos + T, device=idx.device))
        x = tok_emb + pos_emb

        present_kvs = []
        for i, block in enumerate(self.blocks):
            past_i = layer_past[i] if layer_past is not None else None
            x, present_kv = block(x, layer_past=past_i)
            present_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, present_kvs

    @torch.no_grad()
    def generate_kv(self, idx, max_new_tokens):
        self.eval()
        past_key_values = None
        curr_idx = idx

        for _ in range(max_new_tokens):
            input_tok = curr_idx if past_key_values is None else curr_idx[:, -1:]
            logits, _, past_key_values = self(input_tok, layer_past=past_key_values)
            
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            curr_idx = torch.cat((curr_idx, idx_next), dim=1)
            
            if idx_next.item() == 50256: # <|endoftext|> ID
                break

        return curr_idx

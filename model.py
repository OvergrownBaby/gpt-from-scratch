import torch
import torch.nn as nn
from torch.nn import functional as F
from random import randint
import tiktoken

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.Wk = nn.Parameter(torch.randn(d_model, d_k))
        self.Wq = nn.Parameter(torch.randn(d_model, d_k))
        self.Wv = nn.Parameter(torch.randn(d_model, d_k))
        self.d_k = d_k

    def forward(self, X):
        Q = torch.matmul(X, self.Wq)
        V = torch.matmul(X, self.Wv)
        K = torch.matmul(X, self.Wk)

        scores = torch.matmul(Q, K.transpose(-2, -1))/(self.d_k**0.5)
        mask = torch.tril(torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device))
        attention_weights = torch.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
        attention = torch.matmul(attention_weights, V)

        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        # first we find calculate h how many blocks are there
        self.h = int(d_model/d_k)
        self.Wo = nn.Parameter(torch.randn(d_model, d_model))
        self.heads = nn.ModuleList([SelfAttention(d_model, d_k) for _ in range(self.h)])

    def forward(self, X):
        attentions = []
        for head in self.heads:
            attentions.append(head(X))
        attentions = torch.cat(attentions, dim=-1)
        return torch.matmul(attentions, self.Wo)
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_ff = 4 * d_model # ffn's wider dimension
        self.W1 = nn.Parameter(torch.randn(d_model, self.d_ff)) # map to wider dimension
        self.W2 = nn.Parameter(torch.randn(self.d_ff, d_model)) # map back
    
    def forward(self, X):
        hidden  = torch.matmul(X, self.W1)
        hidden = torch.relu(hidden)
        output = torch.matmul(hidden, self.W2)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, d_k:int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, d_k)
        self.ffn = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model:int, max_seq_len:int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, X:list[int]):
        tok = self.tok_emb(X)
        pos_indices = torch.arange(X.shape[-1], device=X.device)
        pos = self.pos_emb(pos_indices)

        return tok + pos
    
class GPT(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, d_k:int, n_layers:int, max_seq_len:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_k = d_k
        self.emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        # first we get a stack of n-transformers
        self.transformers = nn.ModuleList([TransformerBlock(self.d_model, self.d_k) for _ in range(self.n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, X:list[int]):

        # then we first convert indicies into vectors using dict lookup
        vecs = self.emb(X)

        # each transformer block as a layer
        input = vecs
        for block in self.transformers:
            output = block(input)
            input = output

        output = self.ln(output)
        return self.head(output)
        
class LLM():
    def __init__(self, batch_size, sample_len, d_model, d_k, n_layers, lr):
        self.sample_len = sample_len
        self.batch_size = batch_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

        self.model = GPT(vocab_size=self.vocab_size, d_model=d_model, d_k=d_k, n_layers=n_layers, max_seq_len=sample_len)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self, token_batch):
        x_train = torch.tensor([t[:-1] for t in token_batch]).to(self.device)
        y_train = torch.tensor([t[1:] for t in token_batch]).to(self.device)

        logits = self.model(x_train) # (10000, 32, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_train.view(-1))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def generate(self, start, num_tokens):
        output = self.enc.encode(start)
        for i in range(num_tokens):
            logits = self.model(torch.tensor(output[-self.sample_len:]).to(self.device))
            probs = torch.softmax(logits[-1], dim=-1)
            output.append(int(torch.multinomial(probs, 1)))
        return self.enc.decode(output)
    
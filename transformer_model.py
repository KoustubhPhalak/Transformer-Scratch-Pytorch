'''Transformer architecture (with 86M parameters)'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define key parameters
batch_size = 64
d_model = 512
nhead = 8
num_encoder_layers = num_decoder_layers = 4
dim_feedforward = 2048
dropout = 0.1
vocab_size = 37000
max_len = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        seq_len = x.size(1)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        
        # Sequence positions: [0, 1, 2, ..., seq_len - 1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Denominator term: exp(log(1/[10000^(2i/d_model)]))
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() *
            (-math.log(10000.0) / self.d_model)
        )
        
        # sin for [0, 2, 4 ...] positions, cos for [1, 3, 5 ...] positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

class Final_Encoding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # Multiply embedding by sqrt(d_model) as per the original transformer paper
        x = self.embedding(x) * math.sqrt(self.d_model)
        # Add positional encoding
        x = x + self.pos_encoding(x)
        return x

    
# Define single Attention Head
class AttentionHead(nn.Module):
    def __init__(self, head_size, masking=False):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(d_model, head_size) # Query: What the token wants
        self.K = nn.Linear(d_model, head_size) # Key: What the token has
        self.V = nn.Linear(d_model, head_size) # Value: What the token can communicate if other tokens are interested in it
        self.softmax = nn.Softmax(dim=-1)
        self.masking = masking
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x, context=None, padding_mask=None): 
        B, T, C = x.shape
        q = self.Q(x)
        if context is not None: # For cross-attention
            k = self.K(context)
            v = self.V(context)
        else: # For self-attention
            k = self.K(x)
            v = self.V(x)
        attention_weights = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

        if self.masking: # Causal masking for decoder self-attention
            tril = torch.tril(torch.ones(T, T)).to(device)
            attention_weights = attention_weights.masked_fill(tril == 0, float('-inf'))

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ v
    
# Define Multi-Head Attention
class MultiHeadAttention(nn.Module):
    '''Tokens look at each other via Multi-Head Attention'''
    def __init__(self, num_heads, masking=False):
        super(MultiHeadAttention, self).__init__()
        self.head_size = d_model // num_heads
        assert self.head_size * num_heads == d_model, "d_model must be divisible by num_heads"
        self.heads = nn.ModuleList([
            AttentionHead(self.head_size, masking) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context=None, padding_mask=None):
        out = torch.cat([head(x, context, padding_mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Define Feedforward Layer
class FeedForward(nn.Module):
    '''Tokens communicate/talk with each other via FeedForward Layer'''
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x
    
# Define Encoder Layer
class Encoder(nn.Module):
    '''MHA + x(residual connection) -> LN -> FF + x(residual connection) -> LN'''
    def __init__(self, d_model, nhead):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(nhead)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context=None, encoder_padding_mask=None):
        self_attn_out = self.dropout(self.self_attn(x, context, encoder_padding_mask))
        x = self.norm1(x + self_attn_out)
        ff_out = self.dropout(self.ff(x))
        x = self.norm2(x + ff_out)
        return x
    
# Define Decoder Layer
class Decoder(nn.Module):
    '''MHA(self,masked) + x -> LN -> MHA(cross) + x -> LN -> FF + x -> LN'''
    def __init__(self, d_model, nhead, dropout):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadAttention(nhead, masking=True)
        self.cross_attn = MultiHeadAttention(nhead)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, encoder_padding_mask=None, decoder_padding_mask=None):
        self_attn_out = self.dropout(self.self_attn(x, None, decoder_padding_mask))
        x = self.norm1(x + self_attn_out)
        cross_attn_out = self.dropout(self.cross_attn(x, encoder_out, encoder_padding_mask))
        x = self.norm2(x + cross_attn_out)
        ff_out = self.dropout(self.ff(x))
        x = self.norm3(x + ff_out)
        return x

# Define Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, max_len):
        super(Transformer, self).__init__()
        self.encoder_encoding = Final_Encoding(vocab_size, d_model)
        self.decoder_encoding = Final_Encoding(vocab_size, d_model)
        self.encoder = nn.ModuleList([
            Encoder(d_model, nhead) for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            Decoder(d_model, nhead=nhead, dropout=dropout) for _ in range(num_decoder_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, encoder_input, decoder_input, encoder_padding_mask=None, decoder_padding_mask=None):
        encoder_input = self.encoder_encoding(encoder_input)
        decoder_input = self.decoder_encoding(decoder_input)
        encoder_out = encoder_input
        for layer in self.encoder:
            encoder_out = layer(encoder_out, None, encoder_padding_mask)
        for layer in self.decoder:
            decoder_input = layer(decoder_input, encoder_out, encoder_padding_mask, decoder_padding_mask)
        return self.linear(decoder_input)

    def generate(self, encoder_input, encoder_padding_mask=None, max_length=128):
        batch_size = encoder_input.size(0)
        # Initialize decoder input with <sos> token (assuming index 1)
        decoder_input = torch.ones((batch_size, 1), dtype=torch.long, device=device) * 2

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # Run transformer
            logits = self(encoder_input, decoder_input, encoder_padding_mask, None)
            # Get last predicted token
            next_token = logits[:, -1:, :]

            next_token = torch.argmax(next_token, dim=-1)

            # Force finished sequences to keep producing <eos>
            next_token = torch.where(finished.unsqueeze(-1), torch.tensor(3, device=device), next_token)

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=-1)

            # Update finished status (include existing finished sequences)
            finished = finished | (next_token.squeeze(-1) == 3)
            
            # Early exit if all sequences generate <eos> (assuming index 2)
            if finished.all():
                break
        
        return decoder_input[:, 1:]  # Remove initial <sos> token
    
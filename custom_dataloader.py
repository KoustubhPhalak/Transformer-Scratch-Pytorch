import torch
from torch.utils.data import Dataset, DataLoader
from transformer_model import max_len
import torch.nn.functional as F
import sentencepiece as spm

# Create PyTorch Dataset
class WMTDataset(Dataset):
    def __init__(self, de_data, en_data, tokenizer_en, tokenizer_de, max_len=max_len):
        self.de_data = de_data
        self.en_data = en_data
        self.tokenizer_en = tokenizer_en
        self.tokenizer_de = tokenizer_de
        self.max_len = max_len

    def __len__(self):
        return min(len(self.de_data), len(self.en_data))

    def __getitem__(self, idx):
        src = self.tokenizer_de.encode_as_ids(self.de_data[idx])
        tgt = self.tokenizer_en.encode_as_ids(self.en_data[idx])

        # Add special tokens and pad/truncate
        src = [self.tokenizer_de.bos_id()] + src + [self.tokenizer_de.eos_id()]
        tgt = [self.tokenizer_en.bos_id()] + tgt + [self.tokenizer_en.eos_id()]

        # Create decoder input/output pairs. Right shifting is needed to make sure the model predicts the next token
        decoder_input = tgt[:-1]
        decoder_target = tgt[1:]

        if len(src) > self.max_len:
            src = src[:self.max_len]
        if len(decoder_input) > self.max_len:
            decoder_input = decoder_input[:self.max_len]
        if len(decoder_target) > self.max_len:
            decoder_target = decoder_target[:self.max_len]

        return {
            'encoder_input': torch.LongTensor(src),
            'decoder_input': torch.LongTensor(decoder_input),
            'decoder_target': torch.LongTensor(decoder_target),
        }

# Create dataloaders
def collate_fn(batch):
    pad_token = 0
    batched = {
        'encoder_input': [],
        'decoder_input': [],
        'decoder_target': [],
        'encoder_padding_mask': [],
        'decoder_padding_mask': []

    }
    
    for item in batch:
        # Pad encoder input
        enc_pad = max_len - len(item['encoder_input'])
        padded_enc = F.pad(item['encoder_input'], (0, enc_pad), value=pad_token)
        
        # Pad decoder input and target
        dec_pad = max_len - len(item['decoder_input'])
        padded_dec_input = F.pad(item['decoder_input'], (0, dec_pad), value=pad_token)
        padded_dec_target = F.pad(item['decoder_target'], (0, dec_pad), value=pad_token)

        # Create boolean masks for padding (False=real token, True=padding)
        if enc_pad <= 0:
            enc_mask = torch.zeros_like(item['encoder_input'], dtype=torch.bool)[:max_len]
        else:
            enc_mask = torch.cat([
                torch.zeros_like(item['encoder_input'], dtype=torch.bool),
                torch.ones(enc_pad, dtype=torch.bool)
            ])
        
        if dec_pad <= 0:
            dec_mask = torch.zeros_like(item['decoder_input'], dtype=torch.bool)[:max_len]
        else:
            dec_mask = torch.cat([
                torch.zeros_like(item['decoder_input'], dtype=torch.bool),
                torch.ones(dec_pad, dtype=torch.bool)
            ])
        
        batched['encoder_input'].append(padded_enc)
        batched['decoder_input'].append(padded_dec_input)
        batched['decoder_target'].append(padded_dec_target)
        batched['encoder_padding_mask'].append(enc_mask)
        batched['decoder_padding_mask'].append(dec_mask)
    
    return {k: torch.stack(v) for k, v in batched.items()}

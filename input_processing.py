import glob
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from transformer_model import max_len, batch_size
import torch
import torch.nn.functional as F
from utils import build_corpus, get_split
from custom_dataloader import WMTDataset, collate_fn

vocab_size = 37000 # For German-English/English-German translation

# 1. Collect downloaded files (adjust paths as needed)
train_files = {
    'de': 'europarl-v7.de-en.de',
    'en': 'europarl-v7.de-en.en'
}

build_corpus(train_files, 'wmt_corpus_en.txt', lang='en')
build_corpus(train_files, 'wmt_corpus_de.txt', lang='de')

# 3. Train SentencePiece tokenizer (same as original paper)
spm.SentencePieceTrainer.train(
    input='wmt_corpus_en.txt',
    model_prefix='bpe_en',
    vocab_size=vocab_size,
    model_type='bpe',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=['<sep>', '<cls>', '<mask>'],
    input_sentence_size=1000000,  # use subset for faster training
    character_coverage=0.9995,
)

spm.SentencePieceTrainer.train(
    input='wmt_corpus_de.txt',
    model_prefix='bpe_de',
    vocab_size=vocab_size,
    model_type='bpe',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=['<sep>', '<cls>', '<mask>'],
    input_sentence_size=1000000,  # use subset for faster training
    character_coverage=0.9995,
)
    
de_train, de_test = get_split(train_files['de'])
en_train, en_test = get_split(train_files['en'])

# 4. Create tokenizer instance
tokenizer_en, tokenizer_de = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
tokenizer_en.load('bpe_en.model')
tokenizer_de.load('bpe_de.model')


train_dataset = WMTDataset(de_train, en_train, tokenizer_en, tokenizer_de)
test_dataset = WMTDataset(de_test, en_test, tokenizer_en, tokenizer_de)

# Save both train and test loaders for later use
torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')
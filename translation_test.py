'''Test the translation quality of the trained Transformer model to the target language.'''

from transformer_model import *
from custom_dataloader import WMTDataset, collate_fn
from torch.utils.data import DataLoader
import sentencepiece as spm
from sacrebleu import corpus_bleu

# Load the SentencePiece model
tokenizer_de, tokenizer_en = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
tokenizer_de.load('bpe_de.model')
tokenizer_en.load('bpe_en.model')

# Load the test dataset
test_dataset = torch.load('test_dataset.pt')
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Load the model
model = Transformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, max_len).to(device)
model.load_state_dict(torch.load('model/seq_128_d_512_n_4/small_transformer_10.pt', weights_only=True))

# Translate the test dataset

model.eval()
for batch in test_loader:
    encoder_input = batch['encoder_input'].to(device)
    decoder_input = batch['decoder_input'].to(device)
    decoder_target = batch['decoder_target'].to(device)
    encoder_padding_mask = batch['encoder_padding_mask'].to(device)
    decoder_padding_mask = batch['decoder_padding_mask'].to(device)
    
    generation = model.generate(encoder_input, encoder_padding_mask, max_len)
    translations = tokenizer_en.decode_ids(generation.tolist())
    references = tokenizer_en.decode_ids(decoder_input.tolist())
    bleu = corpus_bleu(translations, [references])
    print("********************* TEST SAMPLE *********************")
    print(f'BLEU score: {bleu.score}') # Original paper reports 28.4 BLEU score, so it should be close to this value.
    print(f'ORIGINAL GERMAN: {tokenizer_de.decode_ids(encoder_input[0].tolist())}')
    print(f'GROUND TRUTH ENGLISH: {references[0]}')
    print(f'PREDICTED ENGLISH: {translations[0]}')
    
    del encoder_input, decoder_input, decoder_target, encoder_padding_mask, decoder_padding_mask, generation, translations, references, bleu


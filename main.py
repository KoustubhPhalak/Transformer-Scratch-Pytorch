'''Train transformer model (with 86M parameters) on the WMT dataset'''

from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import glob
import os
from transformer_model import *
from custom_dataloader import WMTDataset, collate_fn
from torch.optim.lr_scheduler import _LRScheduler

class TransformerLRScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1  # Convert to 1-based index
        scale = self.d_model ** -0.5
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
        lr = scale * min(arg1, arg2)
        return [lr for _ in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()

# Define key parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_step = 2
start_epoch = 0
end_epoch = 10
train_batch_end = 15000
test_batch_end = 10

# Load train and test datasets
train_dataset = torch.load('train_dataset.pt')
test_dataset = torch.load('test_dataset.pt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
print('Train batches:', len(train_loader), 'Test batches:', len(test_loader))

# Usage example
tokenizer_en, tokenizer_de = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
tokenizer_de.load('bpe_de.model')
tokenizer_en.load('bpe_en.model')

cnt = 0
for batch in train_loader:
    if cnt == 178: # Random example batch to print
        print(f"Batch input shape: {batch['decoder_input'].shape}")
        print(f"Sample encoder input text: {tokenizer_de.decode_ids(batch['encoder_input'][0].tolist())}")
        print(f"Sample decoder input text: {tokenizer_en.decode_ids(batch['decoder_input'][0].tolist())}")
        break
    cnt += 1

# Define model instance, optimizer and loss function
model = Transformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, max_len).to(device)
opt = torch.optim.Adam(model.parameters(),
                       betas=(0.9, 0.98),
                       eps=1e-9,
                       lr=0.0
                       )
scheduler = TransformerLRScheduler(opt, d_model)
loss_fn = nn.CrossEntropyLoss()

# Load model weights if available
# if os.path.exists(f'model/seq_{max_len}_d_{d_model}_n_{num_encoder_layers}/small_transformer_{start_epoch}.pt'):
#     model.load_state_dict(torch.load(f'model/seq_{max_len}_d_{d_model}_n_{num_encoder_layers}/small_transformer_{start_epoch}.pt'))

# Training loop
for epoch in range(start_epoch, end_epoch):
    tr_loss = 0.
    te_loss = 0.
    model.train()
    for b_num, batch in enumerate(train_loader):
        if b_num == train_batch_end:
            break
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        encoder_padding_mask = batch['encoder_padding_mask'].to(device)
        decoder_padding_mask = batch['decoder_padding_mask'].to(device)
        
        opt.zero_grad()
        output = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

        active_tokens = (decoder_target != tokenizer_de.pad_id()).float()
        loss = (F.cross_entropy(
                output.view(-1, vocab_size), 
                decoder_target.view(-1),
                reduction='none'
            ) * active_tokens.view(-1))
        loss = loss.sum() / active_tokens.sum()
        # loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
        loss.backward()
        tr_loss += loss.item()
        opt.step()
        scheduler.step()
        print(f"Train Batch {b_num+1}/{train_batch_end} Loss: {loss.item()}", end='\r')
    
    model.eval()
    for b_num, batch in enumerate(test_loader):
        if b_num == test_batch_end:
            break
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        encoder_padding_mask = batch['encoder_padding_mask'].to(device)
        decoder_padding_mask = batch['decoder_padding_mask'].to(device)
        
        output = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

        active_tokens = (decoder_target != tokenizer_de.pad_id()).float()
        loss = (F.cross_entropy(
                output.view(-1, vocab_size), 
                decoder_target.view(-1),
                reduction='none'
            ) * active_tokens.view(-1))
        loss = loss.sum() / active_tokens.sum()
        # loss = loss_fn(output.view(-1, vocab_size), decoder_target.view(-1))
        te_loss += loss.item()
        print(f"Test Batch {b_num+1}/{test_batch_end} Loss: {loss.item()}", end='\r')
    
    if (epoch+1) % save_step == 0: # Perform checkpointing
        if not os.path.exists(f'model/seq_{max_len}_d_{d_model}_n_{num_encoder_layers}/'):
            os.makedirs(f'model/seq_{max_len}_d_{d_model}_n_{num_encoder_layers}/')
        torch.save(model.state_dict(), f"model/seq_{max_len}_d_{d_model}_n_{num_encoder_layers}/small_transformer_{epoch+1}.pt")

    print(f"Epoch: {epoch+1}, Train Loss: {tr_loss/train_batch_end}, Test Loss: {te_loss/test_batch_end}")

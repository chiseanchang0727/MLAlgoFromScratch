import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_conifg, get_weight_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the eos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # Build mask for the target (decoder_input), because we don't want the input to watch the future words
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculate the output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # we reuse the output of the encoder for each iteraion of the loop
        
        # Get the probability of the next token by projection layer
        prob = model.project(out[:, -1]) # we only want the porjection of the last token
        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=-1)
        
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=-1)
        
        if next_word == eos_idx:
            break
        
        return decoder_input.squeeze(0)
        

def run_validation(model, validatoin_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    # source_texts = []
    # expected = []
    # predicted = []
    
    # Size of the console window(just use a defalut value)
    console_width = 80
    
    with torch.no_grad(): # Disable the gradient calculation
        for batch in validatoin_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # The batch size of validation dataset is 1
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation."
            
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            # source_text.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)
            
            # Print to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            
            if count == num_examples:
                break
            
    # if writer: # all send to tensorboard
        # TorchMetrics CharErrorRate, BLEU for evaluating translation tasks and the word error rate
            

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    """
    LLM Tokenization concept — Understanding Text Tokenizations for Transformer-based Language Models
    (https://medium.com/@shahrukhx01/llm-tokenizations-understanding-text-tokenizations-for-transformer-based-language-models-a3f50f7c0c16)
    """
    
    # Path: allow us to create absolute path giving relative path
    # config['tokenizer_file] = '../tokenizers/tokenizer_{0}.json
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    
    if not Path.exists(tokenizer_path):
        # If it encounts the unknown words, it will repace it by [UNK]
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        """
        If a word appears at least 2 times in the dataset, it will be included in the vocabulary.
        If a word appears less than 2 times (i.e., it appears only once), 
        it will not be added to the vocabulary and will be treated as an unknown token (UNK or another special token specified).
        """
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_dataset(config):
    # load 'train' part of the dataset
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Train: valid  0.9:0.1
    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])
    
    # Craete the tensor for model
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    valid_ds = BilingualDataset(valid_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_lang_src = 0
    max_lang_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_lang_src = max(max_lang_src, len(src_ids))
        max_lang_tgt = max(max_lang_tgt, len(tgt_ids))
        
    print(f'Max length of source sentence: {max_lang_src}')
    print(f'Max length of target sentence: {max_lang_tgt}')
    
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # With batch_size=1, there’s no need for complex padding masks during validation, making the evaluation straightforward.
    valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=True)
    
    
    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9) # eps: Epsilon
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weight_file_path(config, epoch=config['preload'])
        print(f'Preload the model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1 # Training the model start from the last epoch
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    """
    1. We have to tell the loss_fn what is going to be ignored    
    2. label_smoothing: 
        the model less confident about its decision by distributing the result probability to other tokens, reducing the overfitting and increasing the accuracy 
        0.1 means giving 0.1% of its score to others
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
    
    for epoch in range(initial_epoch, config['num_epoch']):
        
        # tqdm(train_dataloader) is essentially the same as train_dataloader, but it wraps the dataloader with a progress bar for visualization.
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            model.train()
                       
            encoder_input = batch['encoder_input'].to(device)  # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (batch_size, 1, seq_len, seq_len)

            # Run the tensor through transformer
            encoder_input = model.encode(encoder_input, encoder_mask) # (batch_szie, seq_len, d_model)
            decoder_input = model.decode(encoder_input, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            project_output = model.project(decoder_input) # (batch_size, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch_size, seq_len)

            # (batch_size, seq_len, tgt_vocab_size) -> (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) # (batch_size * seq_len)

            # .set_postfix(...) adds or updates a custom message at the end of the progress bar.
            batch_iterator.set_postfix({"loss": f'loss: {loss.item():6.3f}'})

            # Log the loss
            # add_scalar(...) records a single scalar metric (like loss) over time
            writer.add_scalar('train loss', loss.item(), global_step)

            # flush() forces writing everything to disk so that it appears in TensorBoard immediately instead of waiting.
            writer.flush()

            # Backpropagation
            loss.backward()


            """
            Accumulates gradients over multiple iterations before updating the model. 
            This simulates a larger batch size, which can improve model performance.
            Code example:

            accumulation_step = 4
            for idx, batch in enumerate(dataloader):
                loss = model(batch)          
                loss = loss / accumulation_step  # ✅ Scale loss before backpropagation
                loss.backward()  # Compute gradients (but do NOT update optimizer yet)

                if (idx + 1) % accumulation_step == 0:  # ✅ Update every `accumulation_step` iterations
                    optimizer.step()  # Apply accumulated gradients to update weights
                    optimizer.zero_grad()  # Reset gradients
            """
            optimizer.step() #Updates model parameters using gradients
            optimizer.zero_grad() # clear old gradients to avoid accumulation
            
            # usually we can run validation in few steps rather than every batch
            # run_validation(model, valid_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg:batch_iterator.write(msg), global_step, writer)

            global_step += 1

        # Save the model
        model_filename = get_weight_file_path(config, f'{epoch:02d}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        

if __name__  == '__main__':
    config = get_conifg()
    train_model(config)
    gc.collect()
    torch.cuda.empty_cache()

            
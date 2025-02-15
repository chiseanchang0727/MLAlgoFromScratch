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
        trainer = WordLevelTrainer(special_token=['UNK', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_dataset(config):
    # load 'train' part of the dataset
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config['lang_tgt']}', split='train')
    
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
    
    initial_lenght = 0
    global_stop = 0
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
    
    for epoch in range(initial_epoch, config['epoch']):
        model.train()
        
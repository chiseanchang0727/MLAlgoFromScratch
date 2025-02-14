import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    """
    LLM Tokenizations — Understanding Text Tokenizations for Transformer-based Language Models
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
    
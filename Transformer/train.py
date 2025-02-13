import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def get_or_build_tokenizer(config, dataset, language):
    # Path: allow us to create absolute path giving relative path
    # config['tokenizer_file] = '../tokenizers/tokenizer_{0}.json
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        # If it encounts the unknown words, it will repace it by [UNK]
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    return
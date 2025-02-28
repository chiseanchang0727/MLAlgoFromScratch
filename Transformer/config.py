from pathlib import Path

def get_conifg():
    return {
        "batch_size": 8,
        "num_epoch": 30,
        "lr": 1e-5,
        "seq_len": 350, # author had checked, 350 is enough for Italian
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None, # restart training if it is crashed
        "tokenizer_file": "./Transformer/tokenizer_{0}.json",
        "experiment_name": "./Transformer/runs/tmodel",
    }
    
    
def get_weight_file_path(config, epoch: str):
    
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}/{epoch}.pt"
    model_path = Path('.') / 'Transformer' / model_folder / model_basename / f"{epoch}.pt"
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(model_path)
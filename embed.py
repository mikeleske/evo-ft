import torch

from utils_model import load_evo_from_hf
from utils import parse_fasta_file_mimt_gzip
from utils_bio import get_region

import configs.cfg_embed as cfg

import pandas as pd
import numpy as np
from tqdm import tqdm


def get_emb_evo(model, tokenizer, seq):
    global features
    features = {}

    inputs = tokenizer(seq, return_tensors="pt").input_ids.to("cuda")
    _ = model(inputs)
    hidden_states = features['feats'].float() #.numpy(force=True)[0][-1]

    embedding_mean = torch.mean(hidden_states[0], dim=0).cpu()
    return embedding_mean


def get_emb_dnaberts(model, tokenizer, seq):
    inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"].to("cuda")
    hidden_states = model(inputs)[0]
    
    # embedding with mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0).cpu()
    return embedding_mean


def vectorize(model, tokenizer, df: pd.DataFrame, column: str = 'Seq', embeddings_numpy_file: str = None) -> None:
    embeddings_numpy_file = embeddings_numpy_file.replace('REGION', column)

    vectors = list()
    embeddings = None

    for i in tqdm(range(df.shape[0])):
        if 'evo' in cfg.MODEL_ID:
            emb = get_emb_evo(model, tokenizer, df.loc[i, column])
        elif 'DNABERT-S' in cfg.MODEL_ID:
            emb = get_emb_dnaberts(df.loc[i, column])

        vectors.append(emb.reshape(1, emb.shape[0]))

        if len(vectors) == 50:
            vectors = np.vstack(vectors)
            try:
                embeddings = np.load(embeddings_numpy_file)
            except:
                pass

            if isinstance(embeddings, np.ndarray):
                embeddings = np.concatenate((embeddings, vectors), axis=0)
                np.save(embeddings_numpy_file, embeddings)
                print(embeddings.shape)
            else:
                np.save(embeddings_numpy_file, vectors)

            vectors = list()

    # Now write the remaining vectors to the numpy dump
    embeddings = np.load(embeddings_numpy_file)
    vectors = np.vstack(vectors)
    embeddings = np.concatenate((embeddings, vectors), axis=0)
    np.save(embeddings_numpy_file, embeddings)
    print(embeddings.shape)


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def main() -> None:
    global features
    model, tokenizer = load_evo_from_hf(cfg.MODEL_ID)

    if 'evo' in cfg.MODEL_ID:
        model.backbone.blocks[-1].mlp.l3.register_forward_hook(get_features("feats"))

    df = parse_fasta_file_mimt_gzip(
        file=cfg.DATA_CSV,
        domain=cfg.DOMAIN,
    )

    for region in cfg.REGIONS:
        df[region] = df['Seq'].apply(lambda x: get_region(region=region, seq=x))
        print('DataFrame after region processing:', df.shape)

        df = df[df[region] != 'ACGT'].reset_index(drop=True)
        print('DataFrame after cleaning V3V4 misalignment:', df.shape)

    for region in cfg.REGIONS:
        vectorize(model=model, tokenizer=tokenizer, df=df, column=region, embeddings_numpy_file=cfg.EMB_FILE)
    else:
        vectorize(model=model, tokenizer=tokenizer, df=df, column='Seq', embeddings_numpy_file=cfg.EMB_FILE)


# --------------------------------------------------
if __name__ == '__main__':
    main()

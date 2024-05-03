import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import numpy as np

from utils import parse_fasta_file_gtdb_gzip

from tqdm import tqdm



###########################
#
# Load model and tokenizer
#
###########################

model_name = 'togethercomputer/evo-1-131k-base'
#model_name = 'togethercomputer/evo-1-8k-base'
#model_name = 'mikeleske/evo-ft-genus-325'

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model_config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=model_config,
    trust_remote_code=True,
    device_map={"":0},
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "X"



###########################
#
# Auxiliary functions
#
###########################

def get_emb(seq):
    global features
    features = {}

    inputs = tokenizer(seq, return_tensors="pt").input_ids.to("cuda")
    outputs = model(inputs)

    return features['feats'].float().numpy(force=True)[0][-1]


def vectorize(df: pd.DataFrame, dim: int, embeddings_numpy_file: str = None) -> None:
    vectors = list()
    embeddings = None

    for i in tqdm(range(df.shape[0])):
        emb = get_emb(df.loc[i, 'Seq']).reshape(1, dim)
        vectors.append(emb)

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
    print(embeddings.shape)

    return embeddings


###########################
#
# Register hook to get embeddings of last layer
#
###########################

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model.backbone.blocks[-1].mlp.l3.register_forward_hook(get_features("feats"))


###########################
#
# Read data
#
###########################

data_file = './data/GTDB/bac120_ssu_reps_r220.fna.gz'
embeddings_numpy_file = 'bac120_ssu_reps_r220-131K.npy'
domain = 'd__Bacteria'
out_file = 'r220_16S_bac120.csv'
min_seq_length = 1400
max_seq_length = 2000
V3V4_start = 341
V3V4_end   = 785
VDIM = 4096

num_records = 2000

df = parse_fasta_file_gtdb_gzip(
    file=data_file,
    domain=domain,
    out_file=out_file,
    min_seq_length=min_seq_length,
    max_seq_length=max_seq_length,
    #V3V4_start=V3V4_start,
    #V3V4_end=V3V4_end,
    num_records=num_records
)


###########################
#
# Create embeddings
#
###########################

_ = vectorize(df, VDIM, embeddings_numpy_file)
#np.save(embeddings_numpy_file, embeddings)
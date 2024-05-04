import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import numpy as np
from Bio.Seq import Seq

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

def get_primers(start:str = None, end:str = None, seq = None):
    f_primer = None
    r_primer = None
    rev_seq = Seq(seq).reverse_complement()
    
    if start == '341F':
        primer = 'CCTACGGGNGGCWGCAG'

        for W in ['A', 'T']:
            for N in ['A', 'C', 'G', 'T']:
                iter_primer = primer.replace('W',W).replace('N',N)
                if iter_primer in seq:
                    f_primer = iter_primer
                    break
    
    if end == '785R':
        primer = 'GACTACHVGGGTATCTAATCC'

        for H in ['A', 'C', 'T']:
            for V in ['G', 'C', 'A']:
                iter_primer = primer.replace('H',H).replace('V',V)
                if iter_primer in rev_seq:
                    r_primer = iter_primer
                    break

    #print(f_primer, r_primer)
    return (f_primer, r_primer)

def get_region(region:str = None, seq:str = None):
    if region == 'V3V4':
        f_primer, r_primer = get_primers(start='341F', end='785R', seq=seq)
        try:
            return str(Seq(str(Seq(seq.split(f_primer)[1]).reverse_complement()).split(r_primer)[1]).reverse_complement())
        except:
            return str('ACGT')
        
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
region = 'V3V4'
VDIM = 4096

num_records = 2000

df = parse_fasta_file_gtdb_gzip(
    file=data_file,
    domain=domain,
    out_file=out_file,
    min_seq_length=min_seq_length,
    max_seq_length=max_seq_length,
)

if region:
    df['Seq'] = df['Seq'].apply(lambda x: get_region(region=region, seq=x))

print('DataFrame after loading:', df.shape)

df = df[df.Seq != 'ACGT'].reset_index().drop('index')

print('DataFrame after cleaning V3V4 misalignment:', df.shape)


###########################
#
# Create embeddings
#
###########################

_ = vectorize(df, VDIM, embeddings_numpy_file)
#np.save(embeddings_numpy_file, embeddings)
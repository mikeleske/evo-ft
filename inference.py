from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from evo_code.configuration_hyena import StripedHyenaConfig
from evo_code.modeling_hyena import StripedHyenaModelForCausalLM
import torch

import pandas as pd
import random

import configs.cfg_inference as cfg


###########################
#
# Load model and tokenizer
#
###########################

def load_model_and_tokenier():
    #default_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if cfg.LOCAL_PATH:
        model = StripedHyenaModelForCausalLM.from_pretrained(
            cfg.LOCAL_PATH,
            device_map={"":0},
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.LOCAL_PATH, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(
            cfg.MODEL_ID,
            device_map={"":0},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_ID, 
            trust_remote_code=True
        )
    
    tokenizer.pad_token = "~"
    tokenizer.eos_token = "|"

    return model, tokenizer


###########################
#
# Tokenize and generate
#
###########################

def inference(model, tokenizer, seq) -> str:
    prompt = f"{seq}{cfg.PROMPT_SEP}"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        inputs, 
        max_new_tokens=50, 
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.pad_token_id,
        #do_sample=True,
        #top_k=50,
        #top_p=0.95
    )
    return str(tokenizer.batch_decode(outputs, skip_special_tokens=True))


###########################
#
# Process results
#
###########################

def results(results):
    for r in results:
        print(r)

    correct_predictions = sum([r[0] for r in results])
    print(f"{correct_predictions}/{cfg.NUM_INFERENCE} correct.")


def main() -> None:
    model, tokenizer = load_model_and_tokenier()
    df = pd.read_csv(cfg.DATA_CSV, sep=',')

    samples = []
    results = []

    if cfg.NUM_INFERENCE:
        for _ in range(cfg.NUM_INFERENCE):
            row = random.randint(0, df.shape[0]-1)
            samples.append((df.loc[row, cfg.LEVEL], df.loc[row, cfg.COLUMN]))
    else:
        for row in range(df.shape[0]):
            samples.append((df.loc[row, cfg.LEVEL], df.loc[row, cfg.COLUMN]))

    for genus, seq in samples:
        pred  = inference(model, tokenizer, seq).split(cfg.PROMPT_SEP)[1].split('|')[0]
        results.append((genus==pred, genus, pred))
        print(pred)

    results(results)

def infer_single_seq(seq):
    model, tokenizer = load_model_and_tokenier()
    pred  = inference(model, tokenizer, seq).split(cfg.PROMPT_SEP)[1].split('|')[0]
    print(pred)

# --------------------------------------------------
if __name__ == '__main__':
    main()

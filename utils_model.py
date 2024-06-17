import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, BertConfig
from evo_code.modeling_hyena import StripedHyenaModelForCausalLM

def load_evo_from_hf(model_name):
    model_config = AutoConfig.from_pretrained(
        model_name, 
        revision="1.1_fix",
        trust_remote_code=True
    )
    model_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision="1.1_fix",
        config=model_config,
        trust_remote_code=True,
        device_map={"":0},
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    tokenizer.pad_token = "~"
    tokenizer.eos_token = "|"

    return model, tokenizer


def load_evo_from_local(model_name, local_path):
    model = StripedHyenaModelForCausalLM.from_pretrained(
        local_path,
        device_map={"":0},
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    tokenizer.pad_token = "~"
    tokenizer.eos_token = "|"

    return model, tokenizer


def load_dnaberts_from_hf(model_name):
    #config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    config = BertConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name, 
        config=config, 
        trust_remote_code=True, 
        device_map={"":0}
    )

    return model, tokenizer
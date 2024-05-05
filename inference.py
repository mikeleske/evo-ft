from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from evo_code.configuration_hyena import StripedHyenaConfig
from evo_code.modeling_hyena import StripedHyenaModelForCausalLM
import torch

model_id = "togethercomputer/evo-1-131k-base"
default_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)


path = 'sft_evo_genus_131K'

model = StripedHyenaModelForCausalLM.from_pretrained(
    path,
    device_map={"":0},
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = "§"
tokenizer.eos_token = "§"


def inference(seq):
    #prompt = f"### Seq: {row['Seq']}\n ### Genus: "
    prompt = f"{seq}<G>"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_new_tokens=50)#, do_sample=True, top_k=50, top_p=0.95)
    return str(tokenizer.batch_decode(outputs, skip_special_tokens=True))

seq = ''
print(inference(seq))
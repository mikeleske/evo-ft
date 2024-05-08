# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python sft.py \
    --model_name_or_path='togethercomputer/evo-1-131k-base' \
    --learning_rate=1e-5 \
    --weight_decay=0.01 \
    --per_device_train_batch_size=1 \
    --per_device_test_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --dataset_train_name="train" \
    --dataset_test_name="test" \
    --logging_steps=10 \
    --eval_steps=50 \
    --evaluation_strategy="steps"\
    --num_train_epochs=2 \
    --max_seq_length=500 \
    --output_dir="sft_evo_genus_131K-full" \
    --save_safetensors=False \
    --save_only_model=True \
    --save_steps=20000
    
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

#from utils import parse_fasta_file

#import wandb
#wandb.login()

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        #trust_remote_code=model_config.trust_remote_code,
        trust_remote_code=True,
        attn_implementation=model_config.attn_implementation,
        #attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        bf16=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = "|"
    tokenizer.eos_token = "~"

    ################
    # Dataset
    ################

    out_file = 'r220_16S_bac120_sft.csv'

    raw_datasets = load_dataset('csv', data_files=out_file).shuffle(seed=42)
    raw_testvalid = raw_datasets['train'].train_test_split(test_size=0.05)

    train_dataset = raw_testvalid[args.dataset_train_name]
    eval_dataset = raw_testvalid[args.dataset_test_name]

    ################
    # Data Collator
    ################

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['SeqV3V4'])):
            #text = f"### Seq: {example['Seq'][i]}\n ### Genus: {example['Genus'][i]}"
            text = f"{example['SeqV3V4'][i]}<G>{example['Genus'][i]}|"
            output_texts.append(text)
        return output_texts

    response_template = "<G>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            formatting_func=formatting_prompts_func,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            #dataset_text_field=args.dataset_text_field,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=args.packing,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        #trainer.save_model(training_args.output_dir)
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)

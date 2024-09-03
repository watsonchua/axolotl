from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextIteratorStreamer
import torch
from peft import PeftModel
from fire import Fire
from threading import Thread


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, List
import os
import yaml

device = torch.device("cuda")

def generate_with_model(eval_prompt, ft_model, eval_tokenizer, temperature, repetition_penalty, custom_stop_tokens, max_new_tokens):
    streamer = TextIteratorStreamer(eval_tokenizer)
    model_inputs = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, tokenizer=eval_tokenizer, temperature=temperature)
    if custom_stop_tokens is not None:
        generation_kwargs["stop_strings"] = custom_stop_tokens
    thread = Thread(target=ft_model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

#    with torch.no_grad():
#        if custom_stop_tokens is None:
#            model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, temperature=temperature)[0]
#        else:
#            model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, stop_strings=custom_stop_tokens, tokenizer=eval_tokenizer, temperature=temperature)[0]

#        text_output = eval_tokenizer.decode(model_output, skip_special_tokens=False)
#        print(text_output)
#        return text_output



def load_model(cfg, lora_model_dir):
    base_model_id = cfg["base_model"]


    if cfg["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_id,
        quantization_config=bnb_config,  
        trust_remote_code=True,
        token=True
        )
    
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_id,
        load_in_8bit=cfg['load_in_8bit'],
        device_map="auto",
        trust_remote_code=True,
        token=True
        )

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, add_eos_token=False, trust_remote_code=True, use_fast=True)
    # unk token was used as pad token during finetuning, must set the same here
    # eval_tokenizer.pad_token = eval_tokenizer.unk_token # TODO: check which token is used as pad token
    eval_tokenizer.pad_token = cfg["special_tokens"]["pad_token"]
    if lora_model_dir is not None:
        ft_model = PeftModel.from_pretrained(base_model, lora_model_dir)
    else:
        ft_model = base_model


    device = torch.device("cuda")
    print(device)
    ft_model.to(device)
    ft_model.eval()

    print(ft_model)

    return ft_model, eval_tokenizer
# end load model



def main(config_path, lora_model_dir):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    finetuned_model, eval_tokenizer = load_model(cfg, lora_model_dir)

#   custom_stop_tokens = cfg.get("stop_tokens", None)

#    print("Enter/Paste your instruction. Ctrl-D to predict.")

#    while True:
#        contents = []
#        while True:
#            try:
#                line = input()
#            except EOFError:
#                break
#            contents.append(line)
#        prompt = '\n'.join(contents)
#        if not prompt.strip():
#            break
#        output = generate_with_model(eval_prompt=prompt, ft_model=finetuned_model, eval_tokenizer=eval_tokenizer, temperature=0.01, repetition_penalty=1.15, max_new_tokens=500, custom_stop_tokens=custom_stop_tokens)
#        print(output)

    def gradio_generate(prompt):
        yield from generate_with_model(eval_prompt=prompt, ft_model=finetuned_model, eval_tokenizer=eval_tokenizer, temperature=cfg.get("gradio_temperature", 0.01), repetition_penalty=1.15, max_new_tokens=cfg.get("inference_max_new_tokens",500), custom_stop_tokens=cfg.get("stop_strings"))


    import gradio as gr
    
    demo = gr.Interface(
        fn=gradio_generate,
        inputs="textbox",
        outputs="textbox",
        title="Axolotl Gradio Interface",
    )


    demo.queue().launch(
        show_api=True,
        share=cfg.get("gradio_share", False),
        server_name=cfg.get("gradio_server_name", "0.0.0.0"),
        server_port=cfg.get("gradio_server_port", 8501),
    )


if __name__ == "__main__":
    Fire(main)

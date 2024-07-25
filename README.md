# Mistral-Instruct-7B Fine-Tuning for Receipt Information Extraction

## Introduction
This project demonstrates the fine-tuning of the Mistral-7B-Instruct model for extracting key information from receipt images using the SROIE dataset. It showcases advanced NLP techniques, including quantization and Low-Rank Adaptation (LoRA), to achieve efficient and accurate information extraction from OCR text.

## Project Overview
- **Model**: Mistral-7B-Instruct
- **Task**: Information extraction from receipt OCR text
- **Dataset**: SROIE (Scanned Receipts OCR and Information Extraction)
- **Techniques**: 4-bit quantization (NF4), LoRA, Parameter-Efficient Fine-Tuning (PEFT)

## Key Features
- Quantization for efficient GPU usage
- LoRA for parameter-efficient fine-tuning
- Custom data pipeline for OCR text processing
- Advanced training optimizations (gradient checkpointing, mixed-precision training)
- Model evaluation and performance comparison

## Requirements
- Python 3.8+
- PyTorch 1.13+
- Transformers 4.28+
- PEFT 0.3+
- bitsandbytes 0.39+
- accelerate 0.19+

## Installation
```bash
git clone https://github.com/krishnapal2308/mistral_instruct_7b_finetune_sroie.git
cd mistral_instruct_7b_finetune_sroie
pip install -r requirements.txt
```
## Using the Fine-tuned Model

To use the fine-tuned model, you can load it with quantization to reduce memory usage:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load the PEFT configuration and model
peft_model_id = "krishnapal2308/mistral-instruct-7b-finetuned-sroie"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token
```
This code loads the base Mistral-7B-Instruct model with 4-bit quantization, and then applies our fine-tuned LoRA weights. This approach significantly reduces memory requirements while maintaining model performance.
## Example Usage
Simple example of how to use the loaded model:
```python
def generate_response(prompt, model, tokenizer, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = """### Instruction:
Extract the following information from the given OCR text: company, date, address, total.

### Input:
SATS PREMIER LOUNGE SINGAPORE CHANGI AIRPORT TERMINAL 2 DEPARTURE TRANSIT LOUNGE NORTH LEVEL 3 SINGAPORE 819643 TEL: 65822188 TAX INVOICE DATE : 20 APR 2018 TIME: 05:24 PM INV# : 2018042000032950 ITEM AMOUNT ENTRY 1 WALK-IN 64.20 SUB-TOTAL 64.20 GST 7% 4.20 ROUNDING ADJ 0.00 TOTAL 68.40 Goods Sold Are Not Returnable. This is a computer generated receipt. No signature is required.

### Response:
"""

response = generate_response(prompt, model, tokenizer)
```

## Results
* Reduced model size by 48.19% through 4-bit quantization, from 7.24B to 3.75B parameters.
* Decreased the number of trainable parameters by 99.81% using LoRA, from 7.24B to just 13.63M.
* Improved inference speed by approximately 24% while maintaining high accuracy.
* Successfully fine-tuned and deployed the model for specific tasks on Hugging Face.

## Acknowledgements
- Mistral AI for the base Mistral-7B-Instruct model
- The SROIE dataset creators
- Hugging Face for their Transformers and PEFT libraries

import os
import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen-lora-gpu"

# 1. Load Tokenizer and Base Model (in 4-bit to save VRAM)
# Windows-specific: Ensure the accelerator knows we are using CUDA
accelerator = Accelerator(mixed_precision="fp16") # Uses half-precision for speed/memory
device = accelerator.device
model_name = "Qwen/Qwen2.5-7B-Instruct"
# -------------------
# MODEL (Modified for GPU with 4-bit Quantization)
# -------------------
print(f"Loading tokenizer and model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 4-bit quantization config to save VRAM (Required for 7B models on consumer GPUs)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": device}, # Directs model to the specific GPU found by accelerator
    torch_dtype=torch.float16,
)

# 2. Load the LoRA adapters onto the base model
model = PeftModel.from_pretrained(model, adapter_path)
model.gradient_checkpointing_enable()
model.eval()

# 3. Test a prompt
prompt = "### Instruction:\nНапиши связь к слову.\n\n### Input:\nКот.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
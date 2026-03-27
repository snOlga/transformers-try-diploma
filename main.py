from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------
# CONFIG
# -------------------
model_name = "Qwen/Qwen2.5-7B-Instruct"
num_cpu = os.cpu_count() or 3 

# -------------------
# GPU SPEEDUPS
# -------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# -------------------
# LOAD MODEL (4-bit GPU)
# -------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True # Extra memory savings
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    torch_dtype=torch.float16, 
    quantization_config=bnb_config
)

# Prepares the 4-bit model for stable training
model = prepare_model_for_kbit_training(model)

# -------------------
# LORA
# -------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# memory + speed boost
model.gradient_checkpointing_enable()

# -------------------
# DATASET
# -------------------
dataset = load_dataset("json", data_files="/home/olga/hft/data.json")

# Get the names of the original columns to remove them later
column_names = dataset["train"].column_names

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example, num_proc=num_cpu)

def tokenize(example):
    # only tokenize the "text" field
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=False 
    )

tokenized_dataset = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=num_cpu,
    remove_columns=column_names + ["text"] 
)

# -------------------
# COLLATOR
# -------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------
# TRAINING 
# -------------------

training_args = TrainingArguments(
    output_dir="./qwen-lora",
    
    # bigger batch = better GPU usage
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,

    num_train_epochs=3,
    learning_rate=2e-4,

    # logging
    logging_steps=10,
    logging_strategy="steps",

    # no checkpoints (faster)
    save_strategy="no",

    # mixed precision
    fp16=True,

    # CPU workers
    dataloader_num_workers=num_cpu,
    dataloader_pin_memory=True,

    report_to="none",
    optim="paged_adamw_8bit" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # Use the tokenized version
    data_collator=data_collator
)

# -------------------
# COMPILE 
# -------------------
model = torch.compile(model)

# -------------------
# TRAIN
# -------------------
trainer.train()

# -------------------
# SAVE
# -------------------
model.save_pretrained("./qwen-lora")
tokenizer.save_pretrained("./qwen-lora")

print("Finished")
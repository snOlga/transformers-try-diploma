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

def main():
    # Windows-specific: Ensure the accelerator knows we are using CUDA
    accelerator = Accelerator(mixed_precision="bf16") # Uses half-precision for speed/memory
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
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": device}, # Directs model to the specific GPU found by accelerator
        torch_dtype=torch.bfloat16,
    )

    # Necessary for quantized training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, # Increased from 8 for better learning capacity
        lora_alpha=32,
        # Target most linear layers for better fine-tuning quality
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.005,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    # -------------------
    # DATASET
    # -------------------
    # Note: On Windows, num_proc can sometimes cause issues; 
    # if you get a "PicklingError", set num_proc=0
    dataset = load_dataset("json", data_files="real_data.json")
    column_names = dataset["train"].column_names

    def format_example(example):
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        }

    dataset = dataset.map(format_example)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=2048,
            padding=False
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=column_names + ["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8, # Keep small for 7B models unless you have >24GB VRAM
        pin_memory=True # Faster data transfer to GPU
    )

    # -------------------
    # OPTIMIZER
    # -------------------
    # 8-bit Adam optimizer saves more VRAM
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="cosine", # Cosine usually performs better for LLMs
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # -------------------
    # ACCELERATE PREP
    # -------------------
    # Note: We don't prepare the model here because it's already on device via device_map
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    # -------------------
    # TRAIN LOOP
    # -------------------
    print("Starting GPU training...")
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"{datetime.now().strftime('%H:%M:%S')} | Epoch {epoch} | Step {step}/{num_training_steps} | Loss {loss.item():.4f}")

    # -------------------
    # SAVE
    # -------------------
    accelerator.wait_for_everyone()
    # Save the LoRA adapters
    model.save_pretrained("./qwen-lora-gpu")
    tokenizer.save_pretrained("./qwen-lora-gpu")

    print("Finished training. Model saved to ./qwen-lora-gpu")

if __name__ == "__main__":
    main()
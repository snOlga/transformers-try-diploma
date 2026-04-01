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
    os.environ["OPENMP_NUM_THREADS"] = "5"
    num_cpu = 5

    accelerator = Accelerator()
    device = accelerator.device

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # -------------------
    # MODEL (Modified for CPU)
    # -------------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in standard float32 (or bfloat16 if your CPU supports it)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32, 
        # Removed bitsandbytes config
        # Removed device_map (accelerate handles this automatically during prepare)
    )

    # We do not use prepare_model_for_kbit_training on CPU
    # model = prepare_model_for_kbit_training(model) 

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    
    # Gradient checkpointing can save RAM, but is slower. 
    model.gradient_checkpointing_enable()

    # -------------------
    # DATASET
    # -------------------
    dataset = load_dataset("json", data_files="real_data.json")
    column_names = dataset["train"].column_names

    def format_example(example):
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        }

    dataset = dataset.map(format_example, num_proc=num_cpu)

    def tokenize(example):
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=4,
        num_workers=num_cpu,
        pin_memory=True
    )

    # -------------------
    # OPTIMIZER
    # -------------------
    optimizer = AdamW(model.parameters(), lr=2e-4)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    # -------------------
    # ACCELERATE PREP
    # -------------------
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # -------------------
    # TRAIN LOOP
    # -------------------
    print("Starting training...")
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Time {datetime.now()} | Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    # -------------------
    # SAVE
    # -------------------
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained("./qwen-lora", save_function=accelerator.save)
    tokenizer.save_pretrained("./qwen-lora")

    print("Finished training")

if __name__ == "__main__":
    main()
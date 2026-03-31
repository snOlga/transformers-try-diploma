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

def main():
    os.environ["OPENMP_NUM_THREADS"] = "5"
    # -------------------
    # CONFIG
    # -------------------
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    num_cpu = 5 

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

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        quantization_config=bnb_config
    ).cuda()

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

    # Memory + speed boost
    model.gradient_checkpointing_enable()

    # -------------------
    # DATASET
    # -------------------
    data_path = "real_data.json"
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path)

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

    print("Tokenizing dataset...")
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
        log_level = "info",

        # no checkpoints (faster)
        save_strategy="no",

        # mixed precision
        fp16=True,

        # CPU workers
        dataloader_num_workers=num_cpu,
        dataloader_pin_memory=True,

        report_to="none",
        optim="paged_adamw_8bit",

        torch_compile = True,
        # use_cpu = False,
        cp_config = TorchContextParallelConfig(
            cp_comm_strategy="alltoall", 
        )
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"], 
        data_collator=data_collator
    )

    # -------------------
    # COMPILE 
    # -------------------
    # DISABLED FOR WINDOWS
    # model = torch.compile(model) 

    # -------------------
    # TRAIN
    # -------------------
    print("Starting training...")
    print(next(model.parameters()).device)
    trainer.train()

    # -------------------
    # SAVE
    # -------------------
    print("Saving model...")
    model.save_pretrained("./qwen-lora")
    tokenizer.save_pretrained("./qwen-lora")

    print("Finished")

# Mandatory for Windows multiprocessing
if __name__ == "__main__":
    main()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen-lora-gpu"

# 1. Load Tokenizer and Base Model (in 4-bit to save VRAM)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load the LoRA adapters onto the base model
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 3. Test a prompt
prompt = "### Instruction:\nНапиши связь к слову.\n\n### Input:\nКот.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
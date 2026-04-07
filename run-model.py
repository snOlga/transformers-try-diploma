import os
import time
import torch
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import Accelerator

# -------------------
# 1. API DATA MODELS (OpenAI Spec)
# -------------------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen-lora"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

# -------------------
# 2. LOAD MODEL GLOBALLY
# -------------------
base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen-lora-gpu"
accelerator = Accelerator(mixed_precision="bf16") 
device = accelerator.device

print(f"Loading tokenizer and model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Loading in bfloat16. Ensure you have ~15GB VRAM free.
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device, 
    torch_dtype=torch.bfloat16,
)

# Load the LoRA adapters
print("Applying LoRA adapters...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# -------------------
# 3. FASTAPI SETUP
# -------------------
app = FastAPI(title="Local Qwen LoRA API")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Extract the last user message to use as the instruction
        user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        
        # Format it exactly like your training data
        prompt = f"### Instruction:\n{user_message}\n\n### Input:\n\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if request.temperature > 0 else False
            )
        
        # Decode the output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Strip the prompt from the final response so the API only returns the answer
        response_text = full_output.split("### Response:\n")[-1].strip()

        # Format exactly as OpenAI API expects
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                "total_tokens": outputs.shape[1]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------
# 4. RUN SERVER
# -------------------
if __name__ == "__main__":
    print("Starting API server on http://localhost:11434")
    uvicorn.run(app, host="0.0.0.0", port=11434)
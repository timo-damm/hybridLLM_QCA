#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob


# In[ ]:


import warnings
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=TqdmWarning)

from huggingface_hub import login
login("my_auth_key")


# In[ ]:


user_id = "myUID"


# In[ ]:


# Define model paths
developer = "Qwen"
model_name = "Qwen3-32B-AWQ"
model_id = f"{developer}/{model_name}"
model_path = f"/path/to/{user_id}/{developer}/{model_name}"

# Create directory if needed
os.makedirs(model_path, exist_ok=True)


# In[ ]:


# Utility function
# Check if the directory exists and has model files

def is_model_downloaded(path):
    if not os.path.exists(path):
        return False

    # Check for config.json (required for all models)
    if not os.path.exists(os.path.join(path, "config.json")):
        return False

    # Check for any tokenizer file
    has_tokenizer = any(os.path.exists(os.path.join(path, f)) for f in 
                       ["tokenizer.json", "tokenizer_config.json", "vocab.json"])

    # Check for any model file or pattern
    has_model = any(os.path.exists(os.path.join(path, f)) for f in 
                   ["pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"])

    # If no standard model file found, check for sharded files
    if not has_model:
        has_model = len(glob.glob(os.path.join(path, "model-*-of-*.safetensors"))) > 0

    return has_tokenizer and has_model


# In[ ]:


# Download model if it doesn't exist
if not is_model_downloaded(model_path):

    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
    )

    model.save_pretrained(model_path)
    print(f"Model downloaded successfully to {model_path}")
else:
    print(f"Model already exists at {model_path}")

# In[ ]:


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you solve the equation 2x + 3 = 7?"},
] 


# In[ ]:


# # prepare the model input
# prompt = "Give me a very short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False 
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 


# In[ ]:


# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)


# In[ ]:





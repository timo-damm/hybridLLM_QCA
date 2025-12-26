#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pprint


# In[2]:


user_id = "myUID" 


# ## choose the model

# In[3]:


# Define model paths
developer = "Qwen"
model_name = "Qwen3-32B-AWQ" # "Qwen3-30B-A3B" # 
model_id = f"{developer}/{model_name}"
model_path = f"/path/to/file/{user_id}/{developer}/{model_name}"

# Create directory if needed
os.makedirs(model_path, exist_ok=True)


# ## load

# In[4]:


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)


# ## inference

# In[5]:


system_prompt = """ 
You are a helpful qualitative research assistant. The following is a segment of interview or field note data.

Your task is to code the document below according to the codebook provided.
You are working on a research project about repression and burnout in the grassroots activist group [redacted].
Please code thoroughly. If you do not think any of the codes from the codebook are fitting, develop a new code and append it to the document.
In the end, all sentences should be coded.

Only output the coded fragment in JSON format with the keys corresponding to the codes and the value being the fragment.
Do not include any other text or explanation.

"""

def write_prompt(data): 
    return f""" You are a helpful qualitative research assistant. You are working on a research project about repression and burnout in the grassroots activist group [redacted].
    Your task is to code a segment of interview or field note data according to the codebook. 

    Here is the segment of interview or field note data:
    {data}
    Here is the codebook: 
    {codebook}
    While you code the document make sure stick to the concepts defined in the codebook, unless they are really not applicable.
    Here is contextual information to help you understand the document:
    {context}
    Only output the coded fragment in JSON format with the keys corresponding to the codes and the value being the content of the fragment.
    Do not include any other text or explanation.
    """


# In[7]:


def save_model_outputs(interview_name, thinking_content, content, model_name, metrics=None, base_dir="/path/to/dir/"):
    """
    Save model outputs in an organized directory structure with descriptive filenames.

    Args:
        interview_name (str): Name extracted from the interview file
        thinking_content (str): The thinking content from the model
        content (str): The regular content from the model
        model_name (str): Name of the model used
        metrics (dict, optional): Dictionary of metrics to save
        base_dir (str): Base directory for saving outputs
    """
    run_dir = os.path.join(base_dir, f"{interview_name}_summary")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "thinking_content.txt"), "w") as f:
        f.write(thinking_content)

    with open(os.path.join(run_dir, "content.txt"), "w") as f:
        f.write(content)

    if metrics:
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    metadata = {
        "model_name": model_name,
        "interview_name": interview_name,
        "metrics": metrics or {}
    }

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


# In[ ]:


import os
import ast
import json

# === Paths ===
transcripts_dir = "/path/to/dir"
context_path = "/path/to/file"
codebook_path = "/path/to/file"
output_base_dir = "/path/to/dir"

# === Parameters ===
CHUNK_SIZE_WORDS = 1500
MAX_NEW_TOKENS = 2048

# === Load static resources ===
with open(context_path, "r") as f:
    context = f.read()

with open(codebook_path, "r") as f:
    codebook = f.read()

system_prompt = """You are a helpful qualitative research assistant. The following is a segment of interview or field note data.

Your task is to code the document below according to the codebook provided.
You are working on a research project about repression and burnout in the grassroots activist group [redacted].
Please code thoroughly. If you do not think any of the codes from the codebook are fitting, develop a new code and append it to the document.
In the end, all sentences should be coded.

Only output the coded fragment in JSON format with the keys corresponding to the identifier of the codes (e.g. 5.1 or 3.3.2) and the value being the fragment.
Do not include any other text or explanation.
"""

def write_prompt(data_chunk):
    return f"""
You are a helpful qualitative research assistant. You are working on a research project about repression and burnout in the grassroots activist group [redacted].
Your task is to code a segment of interview or field note data according to the codebook. 

Here is the segment of interview or field note data:
{data_chunk}

Here is the codebook: 
{codebook}

While you code the document make sure to stick to the concepts defined in the codebook, unless they are really not applicable.

Here is contextual information to help you understand the document:
{context}

Only output the coded fragment in JSON format with the keys corresponding to the identifier of the codes (e.g. 5.1 or 3.3.2) and the value being the fragment.
Do not include any other text or explanation.
"""

def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# === Main loop ===
for filename in os.listdir(transcripts_dir):
    if not filename.endswith(".md"):
        continue

    interview_name = os.path.splitext(filename)[0]
    interview_path = os.path.join(transcripts_dir, filename)

    try:
        with open(interview_path, "r") as f:
            interview_text = f.read()
    except Exception as e:
        continue

    chunks = chunk_text(interview_text)

    interview_output_dir = os.path.join(output_base_dir, interview_name)
    os.makedirs(interview_output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        print(f"\n🔹 Chunk {idx+1}/{len(chunks)}")

        try:
            prompt = write_prompt(chunk)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=32768).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0  # If no </think>

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

            if content.startswith("```json"):
                content = content.strip()[7:-3].strip()
            elif content.startswith("```"):
                content = content.strip()[3:-3].strip()

            try:
                coded_output = ast.literal_eval(content)
            except Exception as e:
                print(f"failed to parse JSON for {interview_name}, chunk {idx+1}: {e}")
                continue

            chunk_path = os.path.join(interview_output_dir, f"chunk_{idx+1:03d}.json")
            with open(chunk_path, "w") as f:
                json.dump(coded_output, f, indent=2)

        except Exception as e:
            print(f"unexpected error in chunk {idx+1} of {interview_name}: {e}")
            continue


# In[ ]:


import IPython
os._exit(00)


# In[ ]:





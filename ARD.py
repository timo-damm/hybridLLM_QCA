#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pprint


# In[ ]:


user_id = "myUID" 


# ## choose the model

# In[ ]:


# Define model paths
developer = "Qwen"
model_name = "Qwen3-32B-AWQ" # "Qwen3-30B-A3B" # 
model_id = f"{developer}/{model_name}"
model_path = f"/path/to/file{user_id}/{developer}/{model_name}"

# Create directory if needed
os.makedirs(model_path, exist_ok=True)


# ## load

# In[ ]:


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)


# ## inference

# In[ ]:


import os
import ast
import json
import re

# === Paths ===
transcripts_dir = "/path/to/dir"
context_path = "path/to/file"
output_base_dir = "/path/to/dir"

# === Parameters ===
CHUNK_SIZE_WORDS = 1500
MAX_NEW_TOKENS = 2048

# === Load static resources ===
with open(context_path, "r") as f:
    context = f.read()

system_prompt = """You are a helpful qualitative research assistant. The following is a segment of interview or field note data.

Your task is to extract contact data from the document, to create ego networks.
For each document extract the contacts each person had in the first campaign and in the second campaign
For each contact also extract the information about the person. 
If a person mentions they had contact with everyone and the group comprised a forty people, create forty contacts and leave their personal information empty.
If a person says out of these forty people, six were their close friends, assign this label to six of the contacts.
If the person describes a contact, and you made a list based on their numbers of contact before, assign the description to a contact in the list. 
Do not create a new contact.

Only output one JSON object.  
The JSON must have exactly two top-level keys: "campaign_1" and "campaign_2".  
Each should be a dictionary of contacts and their descriptions.  
Do not output multiple JSON objects. Do not include any other text or explanation.

"""

def write_prompt(data_chunk):
    return f""" You are a helpful qualitative research assistant. You are working on a research project about repression and burnout in the grassroots activist group [redacted].
    Your task is to extract contact data from the document, to create ego networks.

    Here is the segment of interview or field note data:
    {data_chunk}

    Here is contextual information to help you understand the document:
    {context}

    Only output one JSON object.  
    The JSON must have exactly two top-level keys: "campaign_1" and "campaign_2".  
    Each should be a dictionary of contacts and their descriptions.  
    Do not output multiple JSON objects. Do not include any other text or explanation.

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
        print(f"could not read {filename}: {e}")
        continue

    chunks = chunk_text(interview_text)
    interview_output_dir = os.path.join(output_base_dir, interview_name)
    os.makedirs(interview_output_dir, exist_ok=True)

    chunks = chunk_text(interview_text)

    interview_output_dir = os.path.join(output_base_dir, interview_name)
    os.makedirs(interview_output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):

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
            model_inputs = tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=32768
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0  # If no </think>

            content = tokenizer.decode(
                output_ids[index:],
                skip_special_tokens=True
            ).strip()

            # Strip code fences if present
            if content.startswith("```json"):
                content = content.strip()[7:-3].strip()
            elif content.startswith("```"):
                content = content.strip()[3:-3].strip()

            try:
                parsed = json.loads(content)

                # Validate structure
                if not ("campaign_1" in parsed and "campaign_2" in parsed):
                    raise ValueError("missing top-level keys")

                chunk_path = os.path.join(
                    interview_output_dir,
                    f"chunk_{idx+1:03d}.json"
                )
                with open(chunk_path, "w") as f:
                    json.dump(parsed, f, indent=2)

            except Exception as e:
                print(f"failed to parse JSON for {interview_name}, chunk {idx+1}: {e}")
                continue

        except Exception as e:
            print(f"❌ unexpected error in chunk {idx+1} of {interview_name}: {e}")
            continue


# In[ ]:


import IPython
os._exit(00)


# ## extracting the data and making networks 
# In[ ]:


import os
import json
from collections import defaultdict

# Path to the top-level folder containing interviews
BASE_DIR = "/path/to/dir"

# Loop over all interview folders
for interview_folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, interview_folder)

    if os.path.isdir(folder_path):
        combined_data = {"campaign_1": defaultdict(list), "campaign_2": defaultdict(list)}

        # Collect all chunk files (sorted for consistency)
        chunk_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".json"))

        for chunk_file in chunk_files:
            chunk_path = os.path.join(folder_path, chunk_file)

            with open(chunk_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)

            # Merge the campaigns separately
            for campaign_key in ["campaign_1", "campaign_2"]:
                if campaign_key in chunk_data:
                    for person, info in chunk_data[campaign_key].items():
                        # Ensure every mention is stored as a dict
                        if isinstance(info, dict):
                            combined_data[campaign_key][person].append(info)
                        elif isinstance(info, str):
                            combined_data[campaign_key][person].append({"description": info})
                        elif isinstance(info, list):
                            for entry in info:
                                if isinstance(entry, dict):
                                    combined_data[campaign_key][person].append(entry)
                                else:
                                    combined_data[campaign_key][person].append({"description": str(entry)})

        # Convert defaultdict back to dict for saving
        combined_data["campaign_1"] = dict(combined_data["campaign_1"])
        combined_data["campaign_2"] = dict(combined_data["campaign_2"])

        # Save the combined file in the same folder as interview_X.json
        output_path = os.path.join(BASE_DIR, f"{interview_folder}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)


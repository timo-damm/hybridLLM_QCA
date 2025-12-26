#!/usr/bin/env python
# coding: utf-8

# # context extraction

# In[2]:


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pprint


# In[3]:


user_id = "myUID"


# ## choose model

# In[4]:


# Define model paths
developer = "Qwen"
model_name = "Qwen3-32B-AWQ" # "Qwen3-30B-A3B" # 
model_id = f"{developer}/{model_name}"
model_path = f"/path/to/file/{user_id}/{developer}/{model_name}"

# Create directory if needed
os.makedirs(model_path, exist_ok=True)


# ## load

# In[5]:


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)


# ## inference

# In[6]:


system_prompt = """ 
You are a helpful qualitative research assistant. The following is a summary of an interview or field notes.

Your task is to extract relevant contextual information.
You are working on a research project about repression and burnout in the grassroots activist group [redacted].
Please extract relevant contextual elements to the campaigns. The campaigns of interest are the [redacted].
Make sure to also extract names, relevant places and definitions of slang words.

Only output the coded fragment in JSON format with the keys corresponding to the codes and the value being the fragment.
Do not include any other text or explanation.

"""

def write_prompt(data): 
    return f""" You are a helpful qualitative research assistant. You are working on a research project about repression and burnout in the grassroots activist group [redacted].
    Your task is to extract relevant contextual information from interview transcripts or field note data. 

    Here is the segment of interview transcript or field note data:
    {data}
    While you extract contextual information from the document make sure to include information that might be relevant for the campaigns even if not directly related..
    Only output the coded fragment in JSON format with the key "summary" and the value being the content of the summary.
    Do not include any other text or explanation..
    """


# In[7]:


def save_model_outputs(interview_name, thinking_content, content, model_name, metrics=None, base_dir="/path/to/dir"):
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
    run_dir = os.path.join(base_dir, f"{interview_name}_context")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "thinking_content.txt"), "w") as f:
        f.write(thinking_content)

    with open(os.path.join(run_dir, "context.txt"), "w") as f:
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
import pprint

summaries_dir = "/path/to/dir"
metrics = {
    "max_new_tokens": 32768
}

for subdir in os.listdir(summaries_dir):
    full_path = os.path.join(summaries_dir, subdir)

    if os.path.isdir(full_path):

        try:
            interview_name = subdir.replace("_summary", "")
            interview_path = os.path.join(full_path, "content.txt")

            if not os.path.isfile(interview_path):
                continue

            with open(interview_path, "r") as f:
                interview = f.read()

            prompt = write_prompt(interview)
            pprint.pprint(prompt)

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
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            try:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=32768
                )
            except Exception as e:
                print(f"model failed on {interview_name}: {e}")
                continue

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            print(thinking_content[:200], "..." if len(thinking_content) > 200 else "")
            print(content[:200], "..." if len(content) > 200 else "")
            
            if content.startswith("```json"):
                content = content.strip()[7:-3].strip() 
            elif content.startswith("```"):
                content = content.strip()[3:-3].strip()

            try:
                summary = ast.literal_eval(content)["summary"]

            except Exception as e:
                print(f"failed to extract summary for {interview_name}: {e}")
                continue

            saved_dir = save_model_outputs(
                interview_name=interview_name,
                thinking_content=thinking_content,
                content=content,
                model_name=model_name,
                metrics=metrics
            )


        except Exception as e:
            print(f"unexpected error in folder {subdir}: {e}")
            continue


# ## shut down kernel

# In[ ]:


import IPython
os._exit(00)


# In[ ]:





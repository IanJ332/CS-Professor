# 1. Import necessary libraries
from transformers import FastLanguageModel, TextStreamer
import torch
import os
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoTokenizer
import time
import numpy as np

# 2. Define the model and tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Load data
EOS_TOKEN = tokenizer.eos_token

def formatting_prompt_func(examples):
    instructions_en = examples["instructions_en"]
    inputs_en = examples["input_en"]
    outputs_en = examples["output_en"]

    # Map to Chinese if instruction is not None
    if "instruction_zh" in examples:
        instructions_ch = examples["instruction_zh"]
        inputs_ch = examples["input_zh"]
        outputs_ch = examples["output_zh"]

        texts = []
        for i, (instr_en, input_en, output_en, instr_ch, input_ch, output_ch) in enumerate(zip(instructions_en,
inputs_en, outputs_en, instructions_ch, inputs_ch, outputs_ch)):
            text = alpaca_prompt.format(instruction=instr_en, input=input_en, output=output_en)

            if instr_ch:
                text += f"\n\n{alpaca_prompt.format(instruction=instr_ch, input=input_ch, output=output_ch)}"

            texts.append(text + EOS_TOKEN)
        return texts

# 4. Map dataset
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese")
dataset = dataset.select(range(100))
dataset = dataset.map(formatting_prompt_func, batched=True)

# 5. Preprocess data
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(list(dataset["input"]), return_tensors="pt", max_len=512)
labels = tokenizer(list(dataset["output"]), return_tensors="pt", max_len=512)

# 6. Train model
model = FastLanguageModel.from_pretrained(model_name, config=None, load_from_cache=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in range(len(inputs["input"])):
        input_ids = inputs["input"][batch].to(device)
        attention_mask = inputs["attention_mask"][batch].to(device)
        labels_ids = labels["output"][batch].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, return_loss=True)
        loss = criterion(outputs, labels_ids)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(inputs['input']):.4f}")

model.eval()
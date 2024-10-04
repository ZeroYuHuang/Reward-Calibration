import os
import sys
import json
import torch
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm, trange
DATA_DIR = "plase set your alpacaeval restuls path here"
BATCH_SIZE=4

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    out = []
    for d in data:
        d["messages"] = [
            {"role": "user", "content": d["instruction"]},
            {"role": "assistant", "content": d["output"]}
        ]
        out.append(d)
    return out
    
if __name__ == "__main__":
    # load the model
    start = float(sys.argv[1])
    end = float(sys.argv[2])
    model_path = "please set your model path here"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # load dataset
    models = os.listdir(DATA_DIR)
    for i, m in tqdm(enumerate(models[::-1])):

        if not start <= i / len(models) < end:
            continue
        if os.path.exists(os.path.join(DATA_DIR, m, "model_outputs_GRM-llama3-8B-distill.json")):
            continue
        print(m)
        dataset = load_json(os.path.join(DATA_DIR, m, "model_outputs.json"))
        scores = []
        for batch_data in tqdm(dataset):
            template_message = tokenizer.apply_chat_template(batch_data["messages"], tokenize=False)
            kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
            tokens = tokenizer.encode_plus(template_message, **kwargs)
            with torch.no_grad():
                # _, _, reward_tensor = model(
                #     tokens["input_ids"].cuda(), 
                #     attention_mask=tokens["attention_mask"].cuda())
                reward_tensor = model(
                    tokens["input_ids"].cuda(), 
                    attention_mask=tokens["attention_mask"].cuda()
                ).logits.reshape(-1)
                score = reward_tensor.cpu().detach().item()
            scores.append(score)
        assert len(dataset) == len(scores)
        scored_dataset = []
        for d, s in zip(dataset, scores):
            sd = {
                "instruction": d["instruction"],
                # "dataset": d["dataset"],
                "output": d["output"],
                "generator": d["generator"],
                "score": s
            }
            scored_dataset.append(sd)

        with open(os.path.join(DATA_DIR, m, "model_outputs_GRM-llama3-8B-distill.json"), 'w') as file:
            json.dump(scored_dataset, file)
    
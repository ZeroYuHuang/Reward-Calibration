import os
import sys
import json
import torch
import copy
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange
DATA_DIR = "plase set your alpacaeval restuls path here"
BATCH_SIZE=1

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


def move_dict_to_cuda(x):
    out = {}
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cuda()
        elif isinstance(v, dict):
            out[k] = move_dict_to_cuda(v)
        else:
            out[k] = v
    return out

if __name__ == "__main__":
    # load the model
    model_path = "please set your model path here"
    model_output_name = "model_outputs_Eurus-RM-7b.json"
    model = AutoModel.from_pretrained(
        model_path, 
        device_map="cuda", 
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # load dataset
    models = os.listdir(DATA_DIR)
    for i, m in tqdm(enumerate(models)):
        if os.path.exists(os.path.join(DATA_DIR, m, model_output_name)):
            continue
        print(m)
        dataset = load_json(os.path.join(DATA_DIR, m, "model_outputs.json"))
        scores = []
        for batch_data in tqdm(dataset):
            text = f"[INST] {batch_data['instruction']} [/INST] {batch_data['output']}"
            input = tokenizer(text, return_tensors="pt")
            input = move_dict_to_cuda(input)
            score = model(**input).item()
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
        print(scored_dataset[0])
        with open(os.path.join(DATA_DIR, m, model_output_name), 'w') as file:
            json.dump(scored_dataset, file)
    
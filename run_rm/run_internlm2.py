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
if __name__ == "__main__":
    OOM_MODLES = [

    ]
    # load the model
    start = float(sys.argv[1])
    end = float(sys.argv[2])

    model_path = "please set your model path here"
    model = AutoModel.from_pretrained(
        model_path, 
        device_map="auto", 
        # load_in_8bit=True,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # load dataset
    models = os.listdir(DATA_DIR)
    for i, m in tqdm(enumerate(models)):
        if m in ["bedrock_claude"]:
            continue
        if not start <= i / len(models) < end:
            continue
        if os.path.exists(os.path.join(DATA_DIR, m, "model_outputs_internlm2-20b-reward.json")):
            continue
        # print(m)
        dataset = load_json(os.path.join(DATA_DIR, m, "model_outputs.json"))
        scores = []
        for batch_id in range(0, len(dataset), BATCH_SIZE):
            batch_data = dataset[batch_id: batch_id + BATCH_SIZE]
            # print([len(d["messages"][0]["content"]) + len(d["messages"][1]["content"]) for d in batch_data])
            batch_chats = [b["messages"] for b in batch_data]
            try:
                batch_scores = model.get_scores(tokenizer, batch_chats)
            except:
                print(f"Errors when running {m}")
                break
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
            torch.cuda.empty_cache()
        if len(dataset) != len(scores):
            continue
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
        # print(scored_dataset[0])
        with open(os.path.join(DATA_DIR, m, "model_outputs_internlm2-20b-reward.json"), 'w') as file:
            json.dump(scored_dataset, file)
    
import os
import sys
import json
import torch
import copy
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm, trange
DATA_DIR = "plase set your alpacaeval restuls path here"

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
    model_path = "please set your model path here"
    model_outputs_name = "model_outputs_Llama-3-OffsetBias-RM-8B.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=model_path,
        device="cuda",
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }

    # load dataset
    models = os.listdir(DATA_DIR)
    for i, m in tqdm(enumerate(models)):
        if os.path.exists(os.path.join(DATA_DIR, m, model_outputs_name)):
            continue
        print(m)
        dataset = load_json(os.path.join(DATA_DIR, m, "model_outputs.json"))
        scores = []
        for batch_data in tqdm(dataset):
            template_message = tokenizer.apply_chat_template(batch_data["messages"], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
            kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
            pipe_outputs = rm_pipe([template_message], **pipe_kwargs)
            score = pipe_outputs[0][0]["score"]
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

        with open(os.path.join(DATA_DIR, m, model_outputs_name), 'w') as file:
            json.dump(scored_dataset, file)

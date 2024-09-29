#  the version_3 of reward calibration on the 
import os
import re
import copy
import bisect
import argparse
import random
import numpy as np
import scipy.stats
import pandas as pd
from transformers import AutoTokenizer
from helpers import print_correlations, print_rb_results
from transformers import HfArgumentParser

DIR = "PATH_TO_REWAERDBENCH_RESULTS"
HF_TOKEN = 'YOUR_HF_TOKEN'
TOKENIZER_MAPPING = {
    "weqweasdas/RM-Gemma-7B-4096": "weqweasdas/RM-Gemma-7B",
    "Nexusflow/Starling-RM-34B" : "01-ai/Yi-34B-Chat",
    "berkeley-nest/Starling-RM-7B-alpha": "meta-llama/Llama-2-7b-chat-hf"
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", required=True)
    # hyper parameters for MeanReward Calibration
    parser.add_argument("--ratio", type=float, default=4)
    parser.add_argument("--min_calibrate_num", type=int, default=20)
    # hyper parameters for LWR calibration
    parser.add_argument("--frac", type=float, default=0.5) # this is parameter for lowss
    return parser

def get_md_features(df):
    def count_markdown_elements(text):
        # Count the number of headers
        header_pattern = re.compile(r'^\s{0,3}#{1,6}\s+', re.MULTILINE)
        headers = len(header_pattern.findall(text))

        # Count the number of list elements (unordered and ordered)
        list_pattern = re.compile(r'^\s*[-+*] |\d+\.\s', re.MULTILINE)
        lists = len(list_pattern.findall(text))

        # Count the number of bold elements
        bold_pattern = re.compile(r'\*\*(.*?)\*\*|__(.*?)__')
        bolds = len(bold_pattern.findall(text))
        return headers+lists+bolds
    
    df["chosen_md_features"] = [count_markdown_elements(t) for t in df["output_chosen"]]
    df["rejected_md_features"] = [count_markdown_elements(t) for t in df["output_rejected"]]
    return df


def get_length_features(df, tokenizer):
    if tokenizer is not None:
        tok_length_fn = lambda x: len(tokenizer.encode(x, max_length=8192))
        df["chosen_prompt_len_tok"] = [tok_length_fn(t) for t in df["prompt"]]
        df["rejected_prompt_len_tok"] = [tok_length_fn(t) for t in df["prompt"]]
        df["chosen_text_len_tok"] = [tok_length_fn(t) for t in df["text_chosen"]]
        df["rejected_text_len_tok"] = [tok_length_fn(t) for t in df["text_rejected"]]
        df["chosen_output_len_tok"] = [tok_length_fn(t) for t in df["output_chosen"]]
        df["rejected_output_len_tok"] = [tok_length_fn(t) for t in df["output_rejected"]]
        
    char_length_fn = lambda x: len(x)
    df["chosen_prompt_len_char"] = [char_length_fn(t) for t in df["prompt"]]
    df["rejected_prompt_len_char"] = [char_length_fn(t) for t in df["prompt"]]
    df["chosen_text_len_char"] = [char_length_fn(t) for t in df["text_chosen"]]
    df["rejected_text_len_char"] = [char_length_fn(t) for t in df["text_rejected"]]
    df["chosen_output_len_char"] = [char_length_fn(t) for t in df["output_chosen"]]
    df["rejected_output_len_char"] = [char_length_fn(t) for t in df["output_rejected"]]
    return df

def get_features(df, tokenizer):
    df = get_length_features(df, tokenizer)
    df = get_md_features(df)
    return df

def print_feature_correlation_with_score(
    df, 
    feature,
    postfix,
    is_return_metrics=True,
):
    features = df[f"chosen_{feature}"].tolist() + df[f"rejected_{feature}"].tolist()
    scores = df["chosen_scores_calibrated"].tolist() + df["rejected_scores_calibrated"].tolist()
    return print_correlations(features, scores, txt=f"{feature}{postfix}", is_return_metrics=is_return_metrics) 

def load_data(args):
    eval_set_path = os.path.join(DIR, "eval-set-scores", f"{args.model}.json")
    print(eval_set_path)
    eval_data = pd.read_json(eval_set_path) if os.path.exists(eval_set_path) else None
    reward_bench_data = pd.read_csv("PATH_TO_REWARDBENCH_DATASET")
    eval_data["prompt"] = reward_bench_data["prompt"]
    eval_data["output_chosen"] = reward_bench_data["chosen"]
    eval_data["output_rejected"] = reward_bench_data["rejected"]
    tmp_data = eval_data
    # some preprocssing steps
    tmp_data["subsets"] = tmp_data["subset"]
    tmp_data["scores_chosen"] = [s[0] if isinstance(s, list) else s for s in tmp_data["scores_chosen"]]
    tmp_data["scores_rejected"] = [s[0] if isinstance(s, list) else s for s in tmp_data["scores_rejected"]]
    
    if args.model in TOKENIZER_MAPPING:
        tokenizer_name = TOKENIZER_MAPPING[args.model]
    else:
        tokenizer_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=HF_TOKEN, trust_remote_code=True)
    tmp_data = get_features(tmp_data, tokenizer)
    tmp_data = tmp_data.drop(columns=["subset", "prompt", "output_chosen", "output_rejected", "text_chosen", "text_rejected"])
    return tmp_data


def find_interval(all_len, all_reward, lower_bound, upper_bound):
    import bisect
    i = bisect.bisect_left(all_len, lower_bound)
    j = bisect.bisect_right(all_len, upper_bound) - 1
    return all_reward[i:j]

    
def calibrate_score_mean_reward(
    df, args, 
    feature_weights=None,
    with_penalty=False
):
    
    res = {}
    def do_calibrate(reward):
        if len(reward) < args.min_calibrate_num:
            return False
        return True
        
    chosen_score_calibrated = df["scores_chosen"] if not with_penalty else df["scores_chosen"] - 0.001 * df["chosen_text_len_char"]
    rejected_score_calibrated = df["scores_rejected"] if not with_penalty else df["scores_rejected"] - 0.001 * df["rejected_text_len_char"]
    df["chosen_score_calibrated"] = chosen_score_calibrated
    df["rejected_score_calibrated"] = rejected_score_calibrated
    for f, fw in feature_weights.items():
        # 1. get Thereshold
        df["delta_f"] = df[f"chosen_{f}"] - df[f"rejected_{f}"]
        df["abs_delta_f"] = [abs(l1 - l2) for l1, l2 in zip(df[f"chosen_{f}"], df[f"rejected_{f}"])]
        calibrate_f = df["abs_delta_f"].mean() /  args.ratio
        feature_score_pairs = []
        for i in range(len(df)):
            d = df.iloc[i]
            feature_score_pairs.append((d[f"chosen_{f}"], chosen_score_calibrated[i]))
            feature_score_pairs.append((d[f"rejected_{f}"], rejected_score_calibrated[i]))
        random.shuffle(feature_score_pairs)
        feature_score_pairs = sorted(feature_score_pairs, key=lambda x: x[0])
        all_features = [f for f, _ in feature_score_pairs]
        all_scores = [r for _, r in feature_score_pairs]
    
        chosen_score_calibrated, rejected_score_calibrated = [], []
        for i in range(len(df)):
            d = df.iloc[i]
            s1, s2 = d["chosen_score_calibrated"], d["rejected_score_calibrated"]
            f1, f2 = d[f"chosen_{f}"], d[f"rejected_{f}"]
            b1 = find_interval(all_features, all_scores, f1 - calibrate_f, f1 + calibrate_f)
            b2 = find_interval(all_features, all_scores, f2 - calibrate_f, f2 + calibrate_f)
            if do_calibrate(b1) and do_calibrate(b2):
                sc1 = s1 - fw * np.mean(b1)
                sc2 = s2 - fw * np.mean(b2)  
            else:
                sc1, sc2 = s1, s2
            chosen_score_calibrated.append(sc1)
            rejected_score_calibrated.append(sc2)
    
        df["chosen_scores_calibrated"] = chosen_score_calibrated
        df["rejected_scores_calibrated"] = rejected_score_calibrated
        df["calibrated_margins"] = df["chosen_scores_calibrated"] - df["rejected_scores_calibrated"]
    df["calibrated_results"] = [1 if m > 0 else 0 for m in df["calibrated_margins"]]
    df["perference_changed"] = [1 if o !=c else 0 for o, c in zip(df["results"], df["calibrated_results"])]
    res  = {"preference_changed": df["perference_changed"].mean()}
    for f in feature_weights.keys():
        res.update(print_feature_correlation_with_score(df, f, postfix=f"_mean_{with_penalty}", is_return_metrics=True))
    res.update(print_rb_results(df, is_return_metrics=True, metric_name="calibrated_results"))
    return res, df

def no_calibration(df, args):
    df["chosen_scores_calibrated"] = df["scores_chosen"]
    df["rejected_scores_calibrated"] = df["scores_rejected"]
    df["margin"] = df["scores_chosen"] - df["scores_rejected"]
    res  = {"preference_changed": 0}
    for f in args.features:
        res.update(print_feature_correlation_with_score(df, f, postfix="", is_return_metrics=True)) 
    res.update(print_rb_results(df, is_return_metrics=True, metric_name="results"))
    return res, df

def calibrate_score_penalty(df, args):
    df["chosen_scores_calibrated"] = df["scores_chosen"] - 0.001 * df["chosen_text_len_char"]
    df["rejected_scores_calibrated"] = df["scores_rejected"] - 0.001 * df["rejected_text_len_char"]
    df["calibrated_margins"] = df["chosen_scores_calibrated"] - df["rejected_scores_calibrated"]
    df["calibrated_results"] = [1 if m > 0 else 0 for m in df["calibrated_margins"]]
    df["perference_changed"] = [1 if o !=c else 0 for o, c in zip(df["results"], df["calibrated_results"])]
    res  = {"preference_changed": df["perference_changed"].mean()}
    for f in args.features:
        res.update(print_feature_correlation_with_score(df, f, postfix="penalty", is_return_metrics=True))
    res.update(print_rb_results(df, is_return_metrics=True, metric_name="calibrated_results"))
    return res, df

def calibrate_score_LWR(
    df, 
    args, 
    features_weights: dict = None,
    with_penalty=False
):
    import bisect
    import statsmodels.api as sm
    chosen_score_calibrated = df["scores_chosen"] if not with_penalty else df["scores_chosen"] - 0.001 * df["chosen_text_len_char"]
    rejected_score_calibrated = df["scores_rejected"] if not with_penalty else df["scores_rejected"] - 0.001 * df["rejected_text_len_char"]
    df["chosen_score_calibrated"] = chosen_score_calibrated
    df["rejected_score_calibrated"] = rejected_score_calibrated
    for f, fw in features_weights.items():
        x_train = df[f"chosen_{f}"].tolist() + df[f"rejected_{f}"].tolist()
        y_train = chosen_score_calibrated.tolist() + rejected_score_calibrated.tolist()
        lowess = sm.nonparametric.lowess(y_train, x_train, frac=args.frac)
        def estimate_bias(x_querys_f, lowess_f):
            y_predict = []
            for x in x_querys_f:
                idx = bisect.bisect_left(lowess_f[:, 0], x)
                y_predict.append(lowess_f[idx, 1])
            return np.array(y_predict)

        chosen_score_calibrated = chosen_score_calibrated - fw * estimate_bias(df[f"chosen_{f}"], lowess)
        rejected_score_calibrated = rejected_score_calibrated - fw * estimate_bias(df[f"rejected_{f}"], lowess) 
    df["chosen_scores_calibrated"] = chosen_score_calibrated
    df["rejected_scores_calibrated"] = rejected_score_calibrated
    df["calibrated_margins"] = chosen_score_calibrated - rejected_score_calibrated
    df["calibrated_results"] = [1 if m > 0 else 0 for m in df["calibrated_margins"]]
    df["perference_changed"] = [1 if o !=c else 0 for o, c in zip(df["results"], df["calibrated_results"])]
    res  = {"preference_changed": df["perference_changed"].mean()}
    for f in features_weights.keys():
        res.update(print_feature_correlation_with_score(df, f, postfix=f"_LWR_{with_penalty}", is_return_metrics=True))
    res.update(print_rb_results(df, is_return_metrics=True, metric_name="calibrated_results"))
    return res, df

def get_feature_weights(args):
    feature_weights = {}
    for f, a in zip(args.features, args.alphas):
        feature_weights[f] = float(a)
    return feature_weights

def get_output_dir(args):
    output_dir = f"results/calibrate_rewardbench/{'_'.join(args.features)}_ratio_{args.ratio}_min_{args.min_calibrate_num}_frac_{args.frac}_alphas_{'_'.join(args.alphas)}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    
    model_name = args.model.split("/")[-1]
    data = load_data(args)
    feature_weights = get_feature_weights(args)
    output_dir = get_output_dir(args)
    
    if not os.path.exists(f"{output_dir}/original_eval_{model_name}.csv"):
        print("Results for original data")
        print(args.model)
        original_res, calibrated_df = no_calibration(copy.deepcopy(data), args)
        df = pd.DataFrame([original_res])
        df.to_csv(f"{output_dir}/original_eval_{model_name}.csv")
        calibrated_df.to_csv(f"{output_dir}/original_eval_data_{model_name}.csv")
    
    if not os.path.exists(f"{output_dir}/penalty_eval_{model_name}.csv"):
        print("Results from length penalty calibration:")
        print(args.model)
        penalty_results, calibrated_df = calibrate_score_penalty(copy.deepcopy(data), args)
        df = pd.DataFrame([penalty_results])
        df.to_csv(f"{output_dir}/penalty_eval_{model_name}.csv") 
        calibrated_df.to_csv(f"{output_dir}/penalty_eval_data_{model_name}.csv")
        
    if not os.path.exists(f"{output_dir}/mean_reward_eval_{model_name}.csv"):
        print("Results for mean reward calibration:")
        print(args.model)
        mean_reward_results, calibrated_df = calibrate_score_mean_reward(copy.deepcopy(data), args, feature_weights=feature_weights)
        df = pd.DataFrame([mean_reward_results])
        df.to_csv(f"{output_dir}/mean_reward_eval_{model_name}.csv")
        calibrated_df.to_csv(f"{output_dir}/mean_reward_eval_data_{model_name}.csv")
        
    if not os.path.exists(f"{output_dir}/LWR_eval_{model_name}.csv"):
        print("Results for LWR calibration:")
        print(args.model)
        if data is None:
            data = load_data(args)
        lwr_results, calibrated_df = calibrate_score_LWR(copy.deepcopy(data), args, features_weights=feature_weights)
        df = pd.DataFrame([lwr_results])
        df.to_csv(f"{output_dir}/LWR_eval_{model_name}.csv")
        calibrated_df.to_csv(f"{output_dir}/LWR_eval_data_{model_name}.csv")
    
    if not os.path.exists(f"{output_dir}/LWR_penalty_eval_{model_name}.csv"):
        print("Results for LWR calibration with penalty for ")
        print(args.model)
        if data is None:
            data = load_data(args)
        lwr_results, calibrated_df = calibrate_score_LWR(copy.deepcopy(data), args, features_weights=feature_weights, with_penalty=True)
        df = pd.DataFrame([lwr_results])
        df.to_csv(f"{output_dir}/LWR_penalty_eval_{model_name}.csv")
        calibrated_df.to_csv(f"{output_dir}/LWR_penalty_eval_data_{model_name}.csv")
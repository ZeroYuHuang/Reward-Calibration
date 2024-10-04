import os
import re
import copy
import json
import argparse
import scipy
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

def print_correlations(arr1, arr2, txt="", is_return_metrics=False):
    if isinstance(arr1, pd.DataFrame):
        arr1 = list(arr1.index)
    if isinstance(arr2, pd.DataFrame):
        arr2 = list(arr2.index)
    s = scipy.stats.spearmanr(arr1, arr2).statistic
    t = scipy.stats.kendalltau(arr1, arr2).statistic

    if is_return_metrics:
        return {f"spearman_{txt}": s, f"kendall_{txt}": t}
    else:
        if txt != "":
            txt = txt + "\n"
        print(f"{txt}Spearman Corr: {s:.3f} || Kendall Corr: {t:.3f}")

HF_TOKEN = ''
DIR = ""
BASELINE = "gpt4_1106_preview"


def get_model_names(model_postfix):
    MODEL_NAMES = []
    # check model_names, ensure that
    # 1) model has all 805 instructions
    # 2) model have weighted_alpaca_eval_gpt4_turbo metrics
    for mn in os.listdir(DIR):
        if not os.path.exists(f"{DIR}/{mn}/weighted_alpaca_eval_gpt4_turbo/annotations.json"):
            continue
        if not os.path.exists(f"{DIR}/{mn}/weighted_alpaca_eval_gpt4_turbo/metrics.json"):
            continue
        if not os.path.exists(f"{DIR}/{mn}/model_outputs.json"):
            continue
        if not os.path.exists(f"{DIR}/{mn}/model_outputs_{model_postfix}.json"):
            continue
        with open(f"{DIR}/{mn}/model_outputs.json", "r") as file:
            data = json.load(file)
        if len(data) != 805:
            continue
        MODEL_NAMES.append(mn)
    return MODEL_NAMES

            
def load_rewarded_outputs(fpath, tokenizer):
    with open(fpath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for d in data:
        message = [
            {"role": "user", "content": d["instruction"]},
            {"role": "assistant", "content": d["output"]}
        ]
        d["text"] = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    df = pd.DataFrame(data)
    df["instruction_id"] = [i for i in range(len(df))]
    df = df[["instruction_id", "generator", "text", "score", "instruction", "output"]].copy()
    if 'yi-large-preview' in fpath:
        df["generator"] = ["yi-large-preview" for _ in range(len(df))]
    elif "Snorkel-Mistral-PairRM-DPO-best-of-16" in fpath:
        df["generator"] = ["Snorkel-Mistral-PairRM-DPO-best-of-16" for _ in range(len(df))]
    elif "Snorkel-Mistral-PairRM-DPO" in fpath:
        df["generator"] = ["Snorkel-Mistral-PairRM-DPO" for _ in range(len(df))] 
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", required=True)
    parser.add_argument("--ratio", type=float, default=4)
    parser.add_argument("--min_calibrate_num", type=int, default=10)
    parser.add_argument("--frac", type=float, default=0.1)
    args = parser.parse_args()
    return args

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
    
    df["md_features"] = [count_markdown_elements(t) for t in df["output"]] 
    return df        


def get_length_features(df, tokenizer):
    if tokenizer is not None:
        tok_length_fn = lambda x: len(tokenizer.encode(x, max_length=8192))
        df["prompt_len_tok"] = [tok_length_fn(t) for t in df["instruction"]]
        df["text_len_tok"] = [tok_length_fn(t) for t in df["text"]]
        df["output_len_tok"] = [tok_length_fn(t) for t in df["output"]]
    
    char_length_fn = lambda x: len(x)
    df["prompt_len_char"] = [char_length_fn(t) for t in df["instruction"]]
    df["text_len_char"] = [char_length_fn(t) for t in df["text"]]
    df["output_len_char"] = [char_length_fn(t) for t in df["output"]]
    return df

def get_tok_length_features(df, tokenizer):
    
    if tokenizer is None:
        return df
    def get_top_tok_len(tok_list):
        tok_lengths = [len(tok) for tok in tok_list]
        sorted_tok_lengths = sorted(tok_lengths)
        return np.mean(tok_lengths[int(len(sorted_tok_lengths) * 0):])

    df["prompt_tok_len"] = [get_top_tok_len(tokenizer.tokenize(t, max_length=8192)) for t in df["instruction"]]
    df["output_tok_len"] = [get_top_tok_len(tokenizer.tokenize(t, max_length=8192)) for t in df["output"]]
    df["text_tok_len"] = [get_top_tok_len(tokenizer.tokenize(t, max_length=8192)) for t in df["text"]]
    return df

def get_features(df, tokenizer):
    df = get_length_features(df, tokenizer)
    df = get_tok_length_features(df, tokenizer)
    df = get_md_features(df)
    return df
        
def load_data(model, model_postfix, features):
    # load the baseline output
    tokenizer = None
    for f in features:
        if 'tok' in f:
            tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=HF_TOKEN, trust_remote_code=True)
            break
    reward_file = os.path.join(DIR, BASELINE, f"model_outputs_{model_postfix}.json")
    base_df = load_rewarded_outputs(reward_file, AutoTokenizer.from_pretrained(model, use_auth_token=HF_TOKEN, trust_remote_code=True))
    base_df = get_features(base_df, tokenizer)
    all_df = []
    MODEL_NAMES = get_model_names(model_postfix)
    for mn in tqdm(MODEL_NAMES):
        reward_file = os.path.join(DIR, mn, f"model_outputs_{model_postfix}.json")
        df = load_rewarded_outputs(reward_file, AutoTokenizer.from_pretrained(model, use_auth_token=HF_TOKEN, trust_remote_code=True))
        df = get_features(df, tokenizer)
        for f in features:
            df[f"base_{f}"] = base_df[f]
        df["base_text_len_char"] = base_df["text_len_char"]
        df["base_score"] = base_df["score"]
        all_df.append(df)
    all_df = pd.concat(all_df)
    all_df = all_df.reset_index().drop("text", axis=1)
    all_df["margins"] = all_df["score"] - all_df["base_score"]
    return all_df


def find_interval(all_len, all_reward, lower_bound, upper_bound):
    import bisect
    i = bisect.bisect_left(all_len, lower_bound)
    j = bisect.bisect_right(all_len, upper_bound) - 1
    return all_reward[i:j]


def calibrate_score_penalty(df):
    df["calibrated_margins"] = df["score"] - df["base_score"] - 0.001 * (df["text_len_char"] - df["base_text_len_char"])
    return df

def calibrate_score_mean_reward(
    df, args,
    feature_weights=None,
    with_penalty=False
):
    
    def do_calibrate(reward, length):
        if len(reward) < args.min_calibrate_num:
            return False
        return True
    
    from tqdm import trange
    from collections import defaultdict
    if with_penalty:
        df["score"] = df["score"] - 0.001 * df["text_len_char"]
        df["base_score"] = df["base_score"] - 0.001 * df["base_text_len_char"]
    df["score_calibrated"] = df["score"]
    df["base_score_calibrated"] = df["base_score"]

    for f, fw in feature_weights.items():
        # 1. get the therehold for mean reward calibration
        df[f"delta_{f}"] = df[f] - df[f"base_{f}"]
        df[f"abs_delta_{f}"] = [abs(f1 - f2) for f1, f2 in zip(df[f], df[f"base_{f}"])]
        calibrate_T = df[f"abs_delta_{f}"].mean() / args.ratio
        feature_score_pairs = []
        for i in trange(len(df)):
            d = df.iloc[i]
            feature_score_pairs.append((d[f], d["score_calibrated"]))
        random.shuffle(feature_score_pairs)
        feature_score_pairs = sorted(feature_score_pairs, key=lambda x: x[0])
        all_feature, all_scores = [f for f, _ in feature_score_pairs], [s for _, s in feature_score_pairs]

        score_calibrated, base_score_calibrated = [], []
        base_score_calibrated = []

        for i in range(len(df)):
            d = df.iloc[i]
            s1, s2 = d["score_calibrated"], d["base_score_calibrated"]
            f1, f2 = d[f], d[f"base_{f}"]
            b1 = find_interval(all_feature, all_scores, f1 - calibrate_T, f1 + calibrate_T)
            b2 = find_interval(all_feature, all_scores, f2 - calibrate_T, f2 + calibrate_T)
            if do_calibrate(b1, f1) and do_calibrate(b2, f2):
                s1 = s1 - fw * np.mean(b1)
                s2 = s2 - fw * np.mean(b2)
            score_calibrated.append(s1)
            base_score_calibrated.append(s2)
        
        df["score_calibrated"] = score_calibrated
        df["base_score_calibrated"] = base_score_calibrated
        df["calibrated_margins"] = df["score_calibrated"] - df["base_score_calibrated"]
    return df

def calibrate_score_LWR(
    df, args,
    feature_weights=None,
    with_penalty=False
):
    import bisect
    if with_penalty:
        df["score"] = df["score"] - 0.001 * df["text_len_char"]
        df["base_score"] = df["base_score"] - 0.001 * df["text_len_char"]
    import statsmodels.api as sm
    score_calibrated = df["score"]
    base_score_calibrated = df["base_score"]
    for f, fw in feature_weights.items():
        print(f"calibrate {f} with {fw}")
        x_train = df[f].tolist()
        # y_train = df["score_calibrated"].tolist()
        y_train = score_calibrated
        lowess = sm.nonparametric.lowess(y_train, x_train, frac=args.frac)
        def estimate_bias(x_query, lowess):
            y_predict = []
            for x in x_query:
                idx = bisect.bisect_left(lowess[:, 0], x)
                y_predict.append(lowess[idx, 1])
            return np.array(y_predict)
        score_calibrated = score_calibrated - fw * estimate_bias(df[f].values, lowess)
        base_score_calibrated = base_score_calibrated - fw * estimate_bias(df[f"base_{f}"].values, lowess)
        df["score_calibrated"] = score_calibrated
        df["base_score_calibrated"] = base_score_calibrated
        df["calibrated_margins"] = df["score_calibrated"] - df["base_score_calibrated"]
    return df


def get_leaderboard(
    df,
    load_model_lc_metrics=None,
):
    df["preference"] = [1 / (1 + np.exp(-m)) for m in df["margins"]]
    df["calibrated_preference"] = [1 / (1 + np.exp(-cm)) for cm in df["calibrated_margins"]]

    df["win"] = [1 if m > 0 else 0 for m in df["margins"]]
    df["calibrated_win"] = [1 if m > 0 else 0 for m in df["calibrated_margins"]]
    leaderboard = []
    for mn in list(df["generator"].unique()):
        temp_dict = {}
        alpace_eval_results = os.path.join(DIR, mn, "weighted_alpaca_eval_gpt4_turbo", "metrics.json")
        if os.path.exists(alpace_eval_results):
            with open(alpace_eval_results) as file:
                metrics = json.load(file)
        else:
            print(f"{mn} does not have alpaca_eval results")
            continue
        df_mn = copy.deepcopy(df[df["generator"] == mn])
        temp_dict["generator"] = mn
        temp_dict["gpt4-raw-winrate"] = metrics["win_rate"]
        temp_dict["gpt4-LC-winrate"] = metrics["length_controlled_winrate"]
        temp_dict["gpt4-discrete-winrate"] = metrics["discrete_win_rate"]
        temp_dict["avg_length"] = df["text_len_char"].mean()
        # select one model
        if load_model_lc_metrics is None:
            temp_dict["rm-winrate"] = df_mn["preference"].mean() * 100
            temp_dict["calibrated-rm-winrate"] = df_mn["calibrated_preference"].mean() * 100
            temp_dict["rm-discrete-winrate"] = df_mn["win"].mean() * 100
            temp_dict["calibrated-rm-discrete-winrate"] = df_mn["calibrated_win"].mean() * 100
        else:
            metric_path = os.path.join(DIR, mn, load_model_lc_metrics, "metrics.json")
            with open(metric_path) as file:
                model_lc_metrics = json.load(file)
            temp_dict["model_win_rate"] = model_lc_metrics["win_rate"]
            temp_dict["calibrated_win_rate"] = model_lc_metrics["length_controlled_winrate"]
        leaderboard.append(temp_dict)

    leaderboard = pd.DataFrame(leaderboard).set_index("generator")
    # leaderboard.to_csv(f"leaderboard_{args.alpha}.csv")
    # leaderboard = pd.read_csv(f"leaderboard_{args.alpha}.csv", index_col=0)
    game_process_v = lambda s : s.replace("_verbose","")
    game_process_c = lambda s : s.replace("_concise","")
    gamed_models = [i for i in leaderboard.index
               if (i + "_verbose") in leaderboard.index and (i + "_concise") in leaderboard.index]
    diff_models = [i for i in leaderboard.index if "_verbose" not in i and i + "_concise" not in i]
    leaderboard["gamed_verbose_only"] = [game_process_v(i) if game_process_v(i) in gamed_models else None for i in leaderboard.index]
    leaderboard["gamed_concise_only"] = [game_process_c(i) if game_process_c(i) in gamed_models else None for i in leaderboard.index]
    return leaderboard, diff_models


def report(lb, metric, diff_models):
        
        output = dict()
        def make_lb_arena_feb(_lb):
            dict_arena = (
                pd.read_csv("data/benchmarks.csv", index_col=0)
                .dropna(subset="LC AlpacaEval 2.0").dropna(subset="Arena Elo\n[Feb 2, 2024]")["Arena Elo\n[Feb 2, 2024]"]
                .squeeze()
            ).to_dict()
            arena_models = [k for k in dict_arena.keys() if k in _lb.index]
            if metric == "gpt4-raw-winrate":
                print(arena_models)
            arena_values = [dict_arena[k] for k in arena_models]
            lb_arena = _lb.loc[arena_models,:]
            lb_arena["ELO"] = arena_values
            
            # lb_arena = lb_arena.dropna()
            return lb_arena
         
        def make_lb_arena_style_control(_lb, style_lb_path):
            dict_arena = (
                pd.read_csv(style_lb_path, index_col=0)["Arena Score"].squeeze()
            ).to_dict()
            arena_models = [k for k in dict_arena.keys() if k in _lb.index]
            arena_models = [k for k in dict_arena.keys() if k in _lb.index]
            if metric == "gpt4-raw-winrate":
                print(arena_models)
            arena_values = [dict_arena[k] for k in arena_models]
            lb_arena = _lb.loc[arena_models,:]
            lb_arena["ELO"] = arena_values
            return lb_arena
        
        lb_arena_feb = make_lb_arena_feb(lb) 
        lb_arena_aug = make_lb_arena_style_control(lb, "data/chatbot_arena_lb_0828.csv")
        lb_arena_length = make_lb_arena_style_control(lb, "data/chatbot_arena_lb_style_length.csv")

    
        # print(f"# Report for **{metric}**")
        # print("## Gameability (lower is better)")
        df_gamed_v = lb.groupby("gamed_verbose_only")[[metric]].agg(["mean","std"]) 
        df_gamed_c = lb.groupby("gamed_concise_only")[[metric]].agg(["mean","std"]) 
        # relative in the sense that models with larger metric shouldn't be considered as having larger vairance
        df_gamed_v[(metric, 'rel_std')] = df_gamed_v[metric]["std"] /  df_gamed_v[metric]["mean"]
        df_gamed_c[(metric, 'rel_std')] = df_gamed_c[metric]["std"] /  df_gamed_c[metric]["mean"] 
        # renormalize to avoid removing gameability by shrinking the scale of the metric
        winrate_std_across_models = lb[lb.index.isin(diff_models)]["gpt4-raw-winrate"].std()
        metric_std_across_models = lb[lb.index.isin(diff_models)][metric].std()
        metric_weight = winrate_std_across_models / metric_std_across_models 
        
        # print(metric_weight)

        verbosity_gameability = df_gamed_v[metric]['rel_std'].mean() * metric_weight * 100
        conciseness_gameability = df_gamed_c[metric]['rel_std'].mean() * metric_weight * 100

        output["Verboisty Gameability"] = verbosity_gameability
        output["Conciseness Gameability"] = conciseness_gameability

        corr = print_correlations(lb[metric], lb["gpt4-LC-winrate"], txt=f"AlpacaEval2", is_return_metrics=True)
        output.update(corr)
        
        corr = print_correlations(lb_arena_feb[metric], lb_arena_feb["ELO"], txt=f"ChatbotArena Feb.", is_return_metrics=True)
        output.update(corr)
        
        corr = print_correlations(lb_arena_aug[metric], lb_arena_aug["ELO"], txt=f"ChatbotArena Aug.", is_return_metrics=True)
        output.update(corr)

        corr = print_correlations(lb_arena_length[metric], lb_arena_length["ELO"], txt=f"ChatbotArena Length", is_return_metrics=True)
        output.update(corr)

        return output

def get_feature_weights(args):
    feature_weights = {}
    for f, a in zip(args.features, args.alphas):
        feature_weights[f] = float(a)
    return feature_weights

def get_output_path(args):
    
    return f"results/alpacaeval_v3/{'_'.join(args.features)}_{args.ratio}_{args.min_calibrate_num}_{args.frac}_{'_'.join(args.alphas)}.csv"
        
        
if __name__ == "__main__":

    MODEL_LIST = [
        ("internlm/internlm2-20b-reward", "internlm2-20b-reward"), # 183 36 40 48 48 48 48 48
        ("NCSOFT/Llama-3-OffsetBias-RM-8B", "Llama-3-OffsetBias-RM-8B"), # 188
        ("internlm/internlm2-7b-reward", "internlm7b"), # 188
        ("Ray2333/GRM-llama3-8B-sftreg", "GRM"), # 188
        ("Ray2333/GRM-llama3-8B-distill", "GRM-llama3-8B-distill"), # 188
        ("sfairXC/FsfairX-LLaMA3-RM-v0.1", "FsfairX-LLaMA3-RM-v0.1"), # 188
        ("CIR-AMS/BTRM_Qwen2_7b_0613", "BTRM_Qwen2_7b_0613"), # 188
        ("openbmb/Eurus-RM-7b", "Eurus-RM-7b"),
        ("weqweasdas/RM-Mistral-7B", "RM-Mistral-7B"),
        ("hendrydong/Mistral-RM-for-RAFT-GSHF-v0", "Mistral-RM-for-RAFT-GSHF-v0"),    
    ]
    
    from copy import deepcopy
    args = get_args()
    feature_weights = get_feature_weights(args)
    output_file = get_output_path(args)
    results = {}

    for model, model_postfix in MODEL_LIST:
        model_name = model.split("/")[-1]
        _df = load_data(model, model_postfix, args.features)
        df = calibrate_score_penalty(deepcopy(_df))
        leaderboard, diff_models = get_leaderboard(df)
        # baseline
        if "gpt4-raw-winrate" not in results:
            results["gpt4-raw-winrate"] = report(leaderboard, "gpt4-raw-winrate", diff_models)
        if "gpt4-LC-winrate" not in results:
            results["gpt4-LC-winrate"] = report(leaderboard, "gpt4-LC-winrate", diff_models)
        
        results[model_name] = report(leaderboard, "rm-winrate", diff_models)
        results[model_name + "discrete"] = report(leaderboard, "rm-discrete-winrate", diff_models)
        
        results[f"{model_name}-Penalty"] = report(leaderboard, "calibrated-rm-winrate", diff_models)
        results[f"{model_name}-Penalty-discrete"] = report(leaderboard, "calibrated-rm-discrete-winrate", diff_models)
        
        # mean reward calibration
        df = calibrate_score_mean_reward(deepcopy(_df), args, feature_weights=feature_weights)
        leaderboard, diff_models = get_leaderboard(df)
        results[f"{model_name}-MeanReward"] = report(leaderboard, "calibrated-rm-winrate", diff_models)
        results[f"{model_name}-MeanReward-discrete"] = report(leaderboard, "calibrated-rm-discrete-winrate", diff_models)
        
        df = calibrate_score_mean_reward(deepcopy(_df), args, feature_weights=feature_weights, with_penalty=True)
        leaderboard, diff_models = get_leaderboard(df)
        results[f"{model_name}-MeanReward"] = report(leaderboard, "calibrated-rm-winrate", diff_models)
        results[f"{model_name}-MeanReward-discrete"] = report(leaderboard, "calibrated-rm-discrete-winrate", diff_models)
        
        # print("length controlled calibration")
        leaderboard, diff_models = get_leaderboard(df, load_model_lc_metrics=model_postfix)
        results[f"{model_name}-LC"] = report(leaderboard, "calibrated_win_rate", diff_models)
        
        # LWR calibration
        df = calibrate_score_LWR(copy.deepcopy(_df), args, feature_weights=feature_weights)
        leaderboard, diff_models = get_leaderboard(df)
        results[f"{model_name}-LOWESS"] = report(leaderboard, "calibrated-rm-winrate", diff_models)
        results[f"{model_name}-LOWESS-discrete"] = report(leaderboard, "calibrated-rm-discrete-winrate", diff_models)  
        
        # LWR calibration with length penalty
        df = calibrate_score_LWR(copy.deepcopy(_df), args, feature_weights=feature_weights, with_penalty=True)
        leaderboard, diff_models = get_leaderboard(df)
        results[f"{model_name}-LOWESS-Penalty"] = report(leaderboard, "calibrated-rm-winrate", diff_models)
        results[f"{model_name}-LOWESS-Penalty-discrete"] = report(leaderboard, "calibrated-rm-discrete-winrate", diff_models)
        pd.DataFrame(results).to_csv(output_file)
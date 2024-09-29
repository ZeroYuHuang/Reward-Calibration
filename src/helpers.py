import scipy
import pandas as pd
import numpy as np

from typing import Union, Optional, Sequence
from alpaca_eval.metrics.glm_winrate import (
    GLM_INFO, get_winrate, _get_featurized_data, fit_LogisticRegressionCV, 
    predict_winrate, get_is_extreme_changes
)
from alpaca_eval import utils

def get_length_controlled_preference_and_delta_len(
    annotations: Union[pd.DataFrame, Sequence[dict]],
    glm_name="length_controlled_v1",
    save_weights_dir = "auto",
    baseline: Optional[str] = None,
    is_add_glm_preference_inplace: bool = True,
    is_warn_extreme_changes: bool = True,
    glm_info=None,
) -> dict[str, float]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference, and also predict the length
    controlled winrate using a GLM.

    Parameters
    ----------
    annotations : pd.DataFrame or Sequence of dict
        The annotations to compute the winrate from.

    glm_name : str, optional
        The name of the GLM to use.

    save_weights_dir : Path, optional
        The directory to save the weights of the GLM. If None, the weights are not saved. If "auto", we save the weights
        weights / annotator / glm_name. Can only be "auto" if there's a unique annotator.

    baseline : str, optional
        The name of the baseline model to compare to. If None, we use the default for that annotation (i.e. output_2).

    is_add_glm_preference_inplace : bool, optional
        Whether to add the GLM preference to the annotations inplace. Only possible if annotations is a DataFrame.

    is_warn_extreme_changes : bool, optional
        Warn if the length controlled win rate is very different from the raw one.

    glm_info : dict, optional
        The information to use for the GLM. If None, we use the default for that glm_name.
    """
    glm_info = glm_info or GLM_INFO[glm_name]

    metrics = get_winrate(annotations)  # get the non-length controlled winrate
    df = utils.convert_to_dataframe(annotations)

    assert len(df["generator_2"].unique()) == 1
    model_name = list(df["generator_2"].unique())[0]
    baseline_name = list(df["generator_1"].unique())[0]
    is_baseline = model_name == baseline_name

    if not is_baseline:
        df_XY_train, df_X_test, sample_weight = _get_featurized_data(
            df,
            formula=glm_info["formula"],
            regularize_to_baseline_lambda=glm_info["regularize_to_baseline_lambda"],
        )
        filter_df = df_XY_train["preference"].notna()
        df_XY_train = df_XY_train[filter_df]
        if sample_weight is not None:
            sample_weight = sample_weight[filter_df]

        model = fit_LogisticRegressionCV(
            df_XY_train, "preference", is_ytrue_proba=True, sample_weight=sample_weight, **glm_info["kwargs"]
        )
        predicted_preferences = model.predict_proba(df_X_test)[:, 1]
        weights = dict(zip(df_X_test.columns, model.coef_[0]))
    else:
        weights = {c.strip(): 0 for c in glm_info["formula"].split("-")[0].split("+")}
        predicted_preferences = (df["preference"] * 0) + 0.5  # by construction

    if is_add_glm_preference_inplace and isinstance(annotations, pd.DataFrame):
        annotations["glm_preference"] = predicted_preferences
    len_1 = df["output_1"].str.len()
    len_2 = df["output_2"].str.len()
    delta_len = len_2 - len_1
    df["preference"] = df["preference"].astype(float).replace({0.0: 1.5}) - 1
    return predicted_preferences, list(delta_len), list(df["preference"])

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores

def print_correlations(arr1, arr2, txt="", is_return_metrics=False):
    if isinstance(arr1, pd.DataFrame):
        arr1 = list(arr1.index)
    if isinstance(arr2, pd.DataFrame):
        arr2 = list(arr2.index)
    s = scipy.stats.spearmanr(arr1, arr2).statistic
    t = scipy.stats.kendalltau(arr1, arr2).statistic

    if is_return_metrics:
        return {
            f"spearman_{txt}": s, 
            # f"kendall_{txt}": t
        }
    else:
        if txt != "":
            txt = txt + "\n"
        # print(f"{txt}Spearman Corr: {s:.3f} || Kendall Corr: {t:.3f}")
        print(f"{txt}Spearman Corr: {s:.3f}")
        
def print_rb_results(out_dataset, is_return_metrics=False, metric_name="results"):
    results_grouped = {}
    present_subsets = np.unique(out_dataset["subsets"])
    for subset in present_subsets:
        # subset_dataset = out_dataset.filter(lambda example: example["subsets"] == subset)
        subset_dataset = out_dataset[out_dataset["subsets"] == subset]
        num_correct = sum(subset_dataset[metric_name])
        num_total = len(subset_dataset[metric_name])
        # print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total
    if "anthropic_helpful" in results_grouped:
        results_leaderboard = {}
        results_leaderboard["pref"] = (results_grouped["anthropic_helpful"] + results_grouped["anthropic_hhh"] + results_grouped["shp"] + results_grouped["summarize"]) / 4
        # pref_dataset = out_dataset[[s in ["anthropic_helpful", "anthropic_hhh", "shp", "summarize"] for s in out_dataset["subsets"]]]
        # results_leaderboard["pref"] = sum(pref_dataset["results"]) / len(pref_dataset["results"])
    else:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
    if is_return_metrics:
        return results_leaderboard
    else:
        print(results_leaderboard)
            
EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond", "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}
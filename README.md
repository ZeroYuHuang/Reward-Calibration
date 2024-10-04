# Post-hoc Reward calibration
This repo contains the code for the paper [**Post-hoc Reward Calibration: A Case Study on Length Bias**](https://arxiv.org/pdf/2409.17407). We propose to use Locally Weighted Regression (LWR) for bias estimation, which is then removed, thereby approximating the underlying true reward. Focusing on the prevalent length bias, we validate the proposed method in three different settings: 

1. Calibrated Reward for RewardBench benchmark.
2. Calibrated Reward for LLMs evaluation.
3. Calibrated Reward for LLMs alignment. 

## Calibrated reward for RewardBench benchmark

You can use the RewardBench setting to understand our method quickly. We calibrate different reward models from the RewardBench leaderboard and re-evaluate them on  RewardBench. A 3.11 performance gain averaged across 33 models is observed:

1. Download the official reward results on RewardBench from https://huggingface.co/datasets/allenai/reward-bench-results. You can use:`git lfs clone https://huggingface.co/datasets/allenai/reward-bench-results `
2. Change the `DIR` and `HF_TOKEN` in `src/calibrate_rewardbench.py`
3. After running `sh calibrate_rewardbench.sh` , the calibrated results are saved in `results/calibrated_rewardbench`
4. To get the figure illustrating the calibration effect, use `notebooks/rewardbench_results.ipynb`

## Calibrated reward for LLMs evaluation

In this paper, based on the AlpacaEval leaderboard, we demonstrate that the calibrated open-source BT-based reward models are not only improved on RewardBench but also have the potential to provide GPT-level evaluation for LLMs. 

1. Download the AlpacaEval results from https://github.com/tatsu-lab/alpaca_eval/tree/main/results
2. Download the reward models you want to calibrate. We provide the individual running scripts for the rm model calibrated in the paper. Please see `run_rm` directory.
3. After running rm on the AlpacaEval benchmark and getting the rewarding results, use `calibrate_alpacaeval.sh` for calibration and `notebook/alpacaeval_results.ipynb` to visualise the results

## Calibrated reward for LLM alignment

We directly use the code from https://github.com/huggingface/alignment-handbook/tree/main for DPO training. We provide the AlpacaEval results in our repo in `results`.

## Citation

If you find this work useful or relevant to your research, please consider citing this paper:

```
@article{huang2024post,
  title={Post-hoc Reward Calibration: A Case Study on Length Bias},
  author={Huang, Zeyu and Qiu, Zihan and Wang, Zili and Ponti, Edoardo M and Titov, Ivan},
  journal={arXiv preprint arXiv:2409.17407},
  year={2024}
}
```




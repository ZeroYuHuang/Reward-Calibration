# Post-hoc Reward calibration
This repo contains the code for the paper **Post-hoc Reward Calibration: A Case Study on Length Bias**. We propose to use Locally Weighted Regression (LWR) for bias estimation, which is then removed, thereby approximating the underlying true reward. Focusing on the prevalent length bias, we validate the proposed method in three different settings: 

1. Calibrated Reward for RewardBench benchmark.
2. Calibrated Reward for LLMs evaluation.
3. Calibrated Reward for LLMs alignment. 

## RewardBench benchmark

You can use the RewardBench setting to quickly understand our method.

1. Download the official reward results on RewardBench from https://huggingface.co/datasets/allenai/reward-bench-results. You can use:`git lfs clone https://huggingface.co/datasets/allenai/reward-bench-results `

2. Change the `DIR` and `HF_TOKEN` in `src/calibrate_rewardbench.py`
3. After running `sh calibrate_rewardbench.sh` , the calibrated results are saved in `results/calibrated_rewardbench`
4. To get the figure illustrating the calibration effect, use `notebooks/rewardbench_results.ipynb`




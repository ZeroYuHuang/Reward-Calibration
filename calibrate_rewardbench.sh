# This is the model list from the RewardBench leaderboard as of Aug. 2024
model_list=(
    'IDEA-CCNL/Ziya-LLaMA-7B-Reward' \
    'NCSOFT/Llama-3-OffsetBias-RM-8B' \
    'openbmb/Eurus-RM-7b' \
    'openbmb/UltraRM-13b' \
    'RLHFlow/RewardModel-Mistral-7B-for-DPA-v1' \
    'internlm/internlm2-7b-reward' \
    'internlm/internlm2-1_8b-reward' \
    'internlm/internlm2-20b-reward' \
    'berkeley-nest/Starling-RM-7B-alpha' \
    'Ray2333/GRM-llama3-8B-sftreg' \
    'Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback' \
    'Ray2333/Gemma-2B-rewardmodel-baseline' \
    'Ray2333/GRM-llama3-8B-distill' \
    'Ray2333/GRM-Gemma-2B-sftreg' \
    'weqweasdas/RM-Gemma-7B' \
    'weqweasdas/RM-Gemma-7B-4096' \
    'weqweasdas/RM-Mistral-7B' \
    'weqweasdas/hh_rlhf_rm_open_llama_3b' \
    'weqweasdas/RM-Gemma-2B' \
    'allenai/tulu-v2.5-70b-preference-mix-rm' \
    'allenai/tulu-v2.5-13b-uf-rm' \
    'allenai/llama-3-tulu-2-8b-uf-mean-rm' \
    'allenai/tulu-v2.5-70b-uf-rm' \
    'allenai/llama-3-tulu-2-70b-uf-mean-rm' \
    'allenai/tulu-v2.5-13b-preference-mix-rm' \
    'PKU-Alignment/beaver-7b-v2.0-cost' \
    'PKU-Alignment/beaver-7b-v1.0-cost' \
    'PKU-Alignment/beaver-7b-v2.0-reward' \
    'PKU-Alignment/beaver-7b-v1.0-reward' \
    'hendrydong/Mistral-RM-for-RAFT-GSHF-v0' \
    'OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1' \
    'OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5' \
    'OpenAssistant/reward-model-deberta-v3-large-v2' \
    'sfairXC/FsfairX-LLaMA3-RM-v0.1' \
    'CIR-AMS/BTRM_Qwen2_7b_0613' \
    'Nexusflow/Starling-RM-34B' \
    'openbmb/MiniCPM-2B-dpo-fp32' \
    'openbmb/Eurus-7b-kt' \
    '0-hero/Matter-0.1-7B-boost-DPO-preview' \
    '0-hero/Matter-0.1-7B-DPO-preview' \
    '0-hero/Matter-0.1-7B-DPO-preview_ref_free' \
    'RLHFlow/LLaMA3-iterative-DPO-final' \
    'ContextualAI/archangel_sft-kto_pythia2-8b' \
    'ContextualAI/archangel_sft-dpo_pythia1-4b' \
    'ContextualAI/archangel_sft-dpo_pythia6-9b' \
    'ContextualAI/archangel_sft-dpo_pythia12-0b_ref_free' \
    'ContextualAI/archangel_sft-kto_pythia12-0b' \
    'ContextualAI/archangel_sft-dpo_llama30b_ref_free' \
    'ContextualAI/archangel_sft-kto_llama13b_ref_free' \
    'ContextualAI/archangel_sft-dpo_llama7b_ref_free' \
    'ContextualAI/archangel_sft-kto_pythia1-4b_ref_free' \
    'ContextualAI/archangel_sft-kto_llama7b_ref_free' \
    'ContextualAI/archangel_sft-dpo_pythia1-4b_ref_free' \
    'ContextualAI/archangel_sft-dpo_llama30b' \
    'ContextualAI/archangel_sft-kto_pythia12-0b_ref_free' \
    'ContextualAI/archangel_sft-kto_pythia6-9b' \
    'ContextualAI/archangel_sft-kto_pythia1-4b' \
    'ContextualAI/archangel_sft-kto_llama13b' \
    'ContextualAI/archangel_sft-kto_llama30b_ref_free' \
    'ContextualAI/archangel_sft-dpo_llama13b_ref_free' \
    'ContextualAI/archangel_sft-dpo_pythia2-8b' \
    'ContextualAI/archangel_sft-dpo_llama7b' \
    'ContextualAI/archangel_sft-kto_llama30b' \
    'ContextualAI/archangel_sft-dpo_pythia12-0b' \
    'ContextualAI/archangel_sft-kto_llama7b' \
    'ContextualAI/archangel_sft-kto_pythia2-8b_ref_free' \
    'ContextualAI/archangel_sft-kto_pythia6-9b_ref_free' \
    'ContextualAI/archangel_sft-dpo_llama13b' \
    'ContextualAI/archangel_sft-dpo_pythia6-9b_ref_free' \
    'ContextualAI/archangel_sft-dpo_pythia2-8b_ref_free' \
    'mistralai/Mixtral-8x7B-Instruct-v0.1' \
    'mistralai/Mixtral-8x7B-Instruct-v0.1_ref_free' \
    'allenai/tulu-2-dpo-70b' \
    'allenai/tulu-2-dpo-13b' \
    'allenai/tulu-2-dpo-7b' \
    'allenai/OLMo-7B-Instruct_ref_free' \
    'allenai/tulu-2-dpo-7b_ref_free' \
    'allenai/OLMo-7B-Instruct' \
    'allenai/tulu-2-dpo-13b_ref_free' \
    'allenai/tulu-2-dpo-70b_ref_free' \
    'allenai/llama-3-tulu-2-dpo-70b' \
    'allenai/llama-3-tulu-2-dpo-8b' \
    'Qwen/Qwen1.5-7B-Chat_ref_free' \
    'Qwen/Qwen1.5-72B-Chat' \
    'Qwen/Qwen1.5-4B-Chat_ref_free' \
    'Qwen/Qwen1.5-72B-Chat_ref_free' \
    'Qwen/Qwen1.5-7B-Chat' \
    'Qwen/Qwen1.5-0.5B-Chat_ref_free' \
    'Qwen/Qwen1.5-1.8B-Chat' \
    'Qwen/Qwen1.5-14B-Chat_ref_free' \
    'Qwen/Qwen1.5-1.8B-Chat_ref_free' \
    'Qwen/Qwen1.5-4B-Chat' \
    'Qwen/Qwen1.5-14B-Chat' \
    'Qwen/Qwen1.5-0.5B-Chat' \
    'Qwen/Qwen1.5-MoE-A2.7B-Chat' \
    'stabilityai/stablelm-zephyr-3b' \
    'stabilityai/stablelm-2-zephyr-1_6b' \
    'stabilityai/stablelm-2-12b-chat' \
    'stabilityai/stablelm-2-zephyr-1_6b_ref_free' \
    'stabilityai/stable-code-instruct-3b' \
    'stabilityai/stablelm-zephyr-3b_ref_free' \
    'wenbopan/Faro-Yi-9B-DPO' \
    'Ahjeong/MMPO_Gemma_7b' \
    'Ahjeong/MMPO_Gemma_7b_gamma1.1_epoch3' \
    'HuggingFaceH4/zephyr-7b-gemma-v0.1_ref_free' \
    'HuggingFaceH4/starchat2-15b-v0.1' \
    'HuggingFaceH4/zephyr-7b-beta' \
    'HuggingFaceH4/zephyr-7b-alpha' \
    'HuggingFaceH4/zephyr-7b-alpha_ref_free' \
    'HuggingFaceH4/zephyr-7b-gemma-v0.1' \
    'HuggingFaceH4/zephyr-7b-beta_ref_free' \
    'jondurbin/bagel-dpo-34b-v0.5' \
    'NousResearch/Nous-Hermes-2-Mistral-7B-DPO_ref_free' \
    'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO_ref_free' \
    'NousResearch/Nous-Hermes-2-Mistral-7B-DPO' \
    'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO' \
    'upstage/SOLAR-10.7B-Instruct-v1.0'
)
for model in "${model_list[@]}"; do
    python src/calibrate_rewardbench.py \
        --model $model \
        --features output_len_char \
        --frac 1.0 \
        --alphas 1.0
done
# llm-uncertainty

---

## Run Experiments

```
python prompt_wip_sel_gpu.py \
  --compute_pre_kl \
  --output_dir output-llama3.1-8b \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --input_file turl_test_2k_prompts_50_unique.jsonl
```

Available Options

```
--model_name	Model ID from Hugging Face or local path
--input_file	Path to the input .jsonl file with prompts
--output_dir	Directory to save experiment results
--temperature	Sampling temperature for generation
--top_p	Nucleus sampling (top-p) value
--max_new_tokens	Maximum number of tokens to generate
--n_repetitions	Number of generations per prompt
--device	Device to run the model on (cuda, cpu, etc.)
--torch_dtype	Data type for model loading (float16, bfloat16, etc.)
--compute_pre_kl	Compute KL divergence over prompt tokens
--topk	Top-K value used in get_topk_dict()
--topn_threshold	Threshold for Top-N filtering in get_topn_dict()
```

---

## Parse Outputs

papermill pipeline-preprocess.ipynb pipeline-preprocess.output.ipynb

---

## Compute Features & Uncertainty Measures

papermill pipeline-measures.ipynb pipeline-measures.output.ipynb

---

## Train Regressor

papermill pipeline-regressor.ipynb pipeline-regressor.output.ipynb


import argparse
import copy
import gc
import json
import math
import os
import pickle
import random
import time

import scipy.stats
import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from accelerate.utils import send_to_device
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM uncertainty analysis using LogitLens approach"
    )

    # Model and data parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        # default="osunlp/TableLlama",
        # default="google/gemma-2-9b-it",
        help="Model ID from HuggingFace or local path",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="turl_test_2k_prompts_50_unique.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--selected_pids_file",
        type=str,
        default="selected_pids.1752.pickle",
        help="Path to pickled selected PIDs file",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument(
        "--max_new_tokens", type=int, default=64, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--n_repetitions", type=int, default=10, help="Number of generations per prompt"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum sequence length. Default is 2^12 + max_new_tokens",
    )

    # Computing parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Computing device",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading",
    )


    args = parser.parse_args()

    # Set default max_tokens if not provided
    if args.max_tokens is None:
        args.max_tokens = 2**12 + args.max_new_tokens

    # Map string dtype to torch dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    args.torch_dtype = dtype_map[args.torch_dtype]

    return args


def generate_prompt(instruction, question, input_seg=None):
    question += " Answer with just a candidate, selected from the provided referent entity candidates list, and nothing else. The selected candidate must be reported verbatim from the list provided as input. Each candidate in the list is enclosed between < and > and reports [DESC] and [TYPE] information."
    if input_seg:
        return PROMPT_DICT["prompt_input"].format(
            instruction=instruction, input_seg=input_seg, question=question
        )
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def flip():
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return


# build prompts
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input_seg}\n\n### Question:\n{question}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

if __name__ == "__main__":
    # Get arguments
    args = parse_args()

    # HuggingFace token
    access_token = None
    if "HF_TOKEN" in os.environ:
        access_token = os.environ["HF_TOKEN"]

    # Set device
    device = torch.device(args.device)

    # Load input data
    with open(args.input_file, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]

    # Load model
    if args.model_name.startswith("osunlp"):
        from llama_attn_replace import replace_llama_attn
        replace_llama_attn()

        # Set RoPE scaling factor
        args.context_size = 8192
        config = transformers.AutoConfig.from_pretrained(args.model_name)

        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and args.context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=args.torch_dtype,
            device_map=device,
        )
        model.resize_token_embeddings(32001)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=(
                args.context_size if args.context_size > orig_ctx_len else orig_ctx_len
            ),
            padding_side="left",
            use_fast=False,
        )
    else:
        config = transformers.AutoConfig.from_pretrained(args.model_name, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=args.torch_dtype, token=access_token
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)
    model.eval()

    # Load selected PIDs
    with open(args.selected_pids_file, "rb") as handle:
        selected_pids = pickle.load(handle)

    NREP = args.n_repetitions
    MAXTOK = args.max_tokens

    outlist = []
    with torch.no_grad():
        for pid, p in enumerate(tqdm(prompts)):
            try:
                start = time.perf_counter()

                # Replace conditional comments with argument checks
                if pid not in selected_pids:
                    continue  # focus on prompts where answer can be wrong

                prompt = generate_prompt(p["instruction"], p["question"], p["input"])
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                p["prompt"] = prompt
                p["tokenized_inputs"] = inputs

                # dirty trick for caching
                # prompt_ = prompt[:-1]
                # inputs_ = tokenizer(prompt_, return_tensors="pt").to(device)

                # ( ! )
                if inputs.input_ids.shape[1] > MAXTOK:
                    continue  # roughly 75% of data kept

                prompt_cache = DynamicCache()

                pre_output = model(**inputs, past_key_values=prompt_cache, use_cache=True)
                prompt_cache = pre_output.past_key_values

                # cleanup
                del pre_output
                flip()

                p["post_output_sequences"] = []
                p["transition_scores_s"] = []
                p["transition_scores_l"] = []

                # NREP generate steps for each prompt
                for kk in range(NREP):
                    generated_block_outputs = {i: [] for i in range(len(model.model.layers) - 1)}

                    # cache crippling
                    cache = copy.deepcopy(prompt_cache)
                    for i in range(len(cache.key_cache)):
                        cache.key_cache[i] = cache.key_cache[i][:, :, :-1, :]
                        cache.value_cache[i] = cache.value_cache[i][:, :, :-1, :]

                    post_output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        output_scores=True,
                        return_dict_in_generate=True,
                        output_logits=True,
                        past_key_values=cache,
                        cache_implementation=None,
                        use_cache=True,
                        do_sample=True,
                    )

                    # sequence
                    post_output_sequences = post_output.sequences.detach().cpu().tolist()
                    p["post_output_sequences"].append(post_output_sequences)

                    # transition scores
                    # https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/models/huggingface_models.py
                    transition_scores_s = model.compute_transition_scores(
                        post_output.sequences, post_output.scores, normalize_logits=True
                    )
                    log_likelihoods_s = [score.item() for score in transition_scores_s[0]]
                    p["transition_scores_s"].append(log_likelihoods_s)
                    transition_scores_l = model.compute_transition_scores(
                        post_output.sequences, post_output.logits, normalize_logits=True
                    )
                    log_likelihoods_l = [score.item() for score in transition_scores_l[0]]
                    p["transition_scores_l"].append(log_likelihoods_l)

                    # cleanup
                    del post_output

                # total processing time
                end = time.perf_counter()
                p["elapsed"] = end - start
                p["args"] = vars(args)
                p["pid"] = pid

                os.makedirs(args.output_dir, exist_ok=True)
                outfile = os.path.join(
                    args.output_dir,
                    f"{args.model_name.split('/')[-1]}.{str(random.randint(0, 2**32))}.pickle",
                )
                with open(outfile, "wb") as handle:
                    pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

                del p
                flip()
            except Exception as e:
                print("EXCEPTION!")
                import traceback

                print(traceback.format_exc())
                continue

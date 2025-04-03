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

    # Uncertainty metrics parameters
    parser.add_argument(
        "--compute_pre_kl", action="store_true", help="Compute KL for prompt tokens"
    )
    parser.add_argument(
        "--compute_pre_iou", action="store_true", help="Compute IOU for prompt tokens"
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-K value for topk dictionary")
    parser.add_argument(
        "--topn_threshold", type=float, default=0.9, help="Threshold for top-N in get_topn_dict"
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


# Update get_topk_dict to use args.topk
def get_topk_dict(logits, k=10):
    # Function remains the same, just using args.topk as default value
    probabilities = torch.softmax(logits, dim=-1)
    top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)
    top_k_probs = torch.gather(probabilities, dim=-1, index=top_k_indices)
    return {"top_k_probs": top_k_probs, "top_k_indices": top_k_indices}


# Update get_topn_dict to use args.topn_threshold
def get_topn_dict(logits, threshold=0.9):
    # Function remains the same, just using args.topn_threshold as default value
    probabilities = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # cutoff
    mask = cumulative_probs >= threshold
    top_n_lengths = mask.int().argmax(dim=-1) + 1

    # gather required only
    top_n_values = torch.gather(logits, dim=-1, index=sorted_indices)
    top_n_probs = sorted_probs
    top_n_indices = sorted_indices

    batched_results = []
    for i in range(logits.shape[1]):
        n = top_n_lengths[0][i].item()
        batched_results.append(
            {
                "top_n_probs": top_n_probs[0, i, :n].to(torch.float32),  # .cpu().tolist(),
                "top_n_indices": top_n_indices[0, i, :n].to(torch.int),
            }
        )

    return batched_results


def compute_entropy_scipy(logits):
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().float().numpy()
    entropy_values = [scipy.stats.entropy(row).tolist() for row in probabilities[0]]
    return entropy_values

def compute_entropy_scipy_device(logits):
    probabilities = torch.softmax(pre_output.logits, dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    return entropy[0].detach().cpu().tolist()


# processes data on device, returns python list in ram
def get_layers_kl_div_mod(pre_output, model, precomp=None):
    # pre_output: B x P x V
    # precomp: L x B x P x D
    kls = []
    softmaxed_log = torch.log_softmax(pre_output, dim=-1)
    for k, l in enumerate(model.model.layers):
        if k == len(model.model.layers) - 1:
            break

        # recover normalized from precomputed (if generated / hooks), or just read from network state
        l_ = model.lm_head(model.model.norm(precomp[k])).detach()  # B x P x V
        softmaxed_l = torch.log_softmax(l_, dim=-1)
        p = softmaxed_l[0]  # P x V
        q = softmaxed_log[0]  # P x V
        kl = torch.nn.functional.kl_div(p, q, reduction="none", log_target=True).sum(dim=1)  # P
        kls.append(kl.detach().cpu().tolist())
    return kls


def jaccard_similarity(list1, list2):
    intersection = len(set(list1) & set(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return float(intersection) / union if union != 0 else 0.0


def get_layers_iou_div_mod(pre_output_proba_topn, model, precomp=None, device="cuda"):
    ious = []
    topn_full = torch.zeros(
        len(pre_output_proba_topn), model.lm_head.weight.shape[0], device=device, dtype=torch.bool
    )  # P x V
    full_indices = [
        (r, i.item())
        for r, topn in enumerate(pre_output_proba_topn)
        for i in topn["top_n_indices"]
    ]
    #print("DIM", topn_full.shape)
    #print("INDICES", full_indices)
    full_indices = torch.tensor(full_indices).t()
    topn_full[full_indices[0], full_indices[1]] = 1

    # for each layer
    for k, l in enumerate(model.model.layers):
        if k == len(model.model.layers) - 1:
            break

        # recover normalized from precomputed (if generated / hooks), or just read from network state
        l_ = model.lm_head(model.model.norm(precomp[k])).detach()
        layer_topn = get_topn_dict(l_)
        layer_topn_full = torch.zeros(
            len(layer_topn), model.lm_head.weight.shape[0], device=device, dtype=torch.bool
        )  # P x V
        full_indices = [
            (r, i.item()) for r, topn in enumerate(layer_topn) for i in topn["top_n_indices"]
        ]
        full_indices = torch.tensor(full_indices).t()
        layer_topn_full[full_indices[0], full_indices[1]] = 1

        iou = torch.logical_and(topn_full, layer_topn_full).sum(dim=1) / torch.logical_or(
            topn_full, layer_topn_full
        ).sum(dim=1)
        ious.append(iou.detach().cpu().tolist())

        # # for each token
        # layer_ious = []
        # for a, b in zip(pre_output_proba_topn, layer_topn):
        #     a = a["top_n_indices"]
        #     b = b["top_n_indices"]
        #     layer_ious.append(jaccard_similarity(a, b))
        # ious.append(layer_ious)
    return ious


def hook_fn(layer_idx, gen, in_generate=False):
    def hook(module, input, output):
        if isinstance(output, tuple) and hasattr(module, "block_output"):
            clo = module.block_output.clone().detach()
            if in_generate and clo.shape[1] > 1:
                # first token returns all prompt tokens, patch last
                clo = clo[:, -1:, :]
            gen[layer_idx].append(clo)

    return hook


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.block_output = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output = output[0]
        return output


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

    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = BlockOutputWrapper(layer)

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
                # attach hooks if needed
                if args.compute_pre_kl or args.compute_pre_iou:
                    generated_block_pre = {i: [] for i in range(len(model.model.layers) - 1)}
                    hooks = []
                    for i, layer in enumerate(model.model.layers):
                        if i == len(model.model.layers) - 1:
                            break
                        hook = layer.register_forward_hook(
                            hook_fn(i, generated_block_pre, in_generate=False)
                        )
                        hooks.append(hook)

                pre_output = model(**inputs, past_key_values=prompt_cache, use_cache=True)
                prompt_cache = pre_output.past_key_values

                # detach hooks if needed
                if args.compute_pre_kl or args.compute_pre_iou:
                    for hook in hooks:
                        hook.remove()

                    # accumulate norms at each generated token
                    for i in generated_block_pre:
                        generated_block_pre[i] = torch.cat(
                            generated_block_pre[i], dim=0
                        )  # B=1 x P x D
                    generated_block_pre = list(generated_block_pre.values())

                # top-n + top-k
                p["pre_output_proba_topn"] = get_topn_dict(
                    pre_output.logits, threshold=args.topn_threshold
                )
                p["pre_output_proba_topk"] = get_topk_dict(pre_output.logits, k=args.topk)
                p["pre_output_true_entropies"] = compute_entropy_scipy(pre_output.logits)

                # Add conditional execution of KL/IOU calculations based on args (compute before detaching)
                if args.compute_pre_kl:
                    p["pre_output_layers_kl"] = get_layers_kl_div_mod(
                        pre_output.logits, model, generated_block_pre
                    )

                if args.compute_pre_iou:
                    p["pre_output_layers_iou"] = get_layers_iou_div_mod(
                        p["pre_output_proba_topn"], model, generated_block_pre
                    )

                # to python lists
                for ty in p["pre_output_proba_topn"]:
                    ty["top_n_probs"] = ty["top_n_probs"].detach().cpu().tolist()
                    ty["top_n_indices"] = ty["top_n_indices"].detach().cpu().tolist()
                p["pre_output_proba_topk"]["top_k_probs"] = (
                    p["pre_output_proba_topk"]["top_k_probs"].detach().cpu().tolist()
                )
                p["pre_output_proba_topk"]["top_k_indices"] = (
                    p["pre_output_proba_topk"]["top_k_indices"].detach().cpu().tolist()
                )

                # cleanup
                del pre_output
                flip()

                p["post_output_sequences"] = []
                p["post_output_proba_topn"] = []
                p["post_output_proba_topk"] = []
                p["post_output_true_entropies"] = []
                p["post_output_layers_kl"] = []
                p["post_output_layers_iou"] = []
                p["transition_scores_s"] = []
                p["transition_scores_l"] = []

                # NREP generate steps for each prompt
                for kk in range(NREP):
                    generated_block_outputs = {i: [] for i in range(len(model.model.layers) - 1)}

                    # register hooks
                    hooks = []

                    for i, layer in enumerate(model.model.layers):
                        if i == len(model.model.layers) - 1:
                            break
                        hook = layer.register_forward_hook(
                            hook_fn(i, generated_block_outputs, in_generate=True)
                        )
                        hooks.append(hook)

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

                    # remove hooks
                    for hook in hooks:
                        hook.remove()

                    # accumulate norms at each generated token
                    for i in generated_block_outputs:
                        # B=1 x P x D
                        generated_block_outputs[i] = torch.cat(
                            generated_block_outputs[i], dim=0
                        ).transpose(0, 1)
                    generated_block_outputs = list(generated_block_outputs.values())

                    # sequence
                    post_output_sequences = post_output.sequences.detach().cpu().tolist()
                    p["post_output_sequences"].append(post_output_sequences)

                    # Cat output logits
                    post_output_scores = torch.cat(post_output.logits, dim=0).unsqueeze(
                        0
                    )  # B=1 x P x V

                    # top-n + top-k
                    mypost_topn = get_topn_dict(post_output_scores)
                    mypost_topk = get_topk_dict(post_output_scores)

                    p["post_output_true_entropies"].append(
                        compute_entropy_scipy(post_output_scores)
                    )

                    # logit lens KL/IOU
                    p["post_output_layers_kl"].append(
                        get_layers_kl_div_mod(post_output_scores, model, generated_block_outputs)
                    )
                    # p["post_output_layers_iou"].append(
                    #     get_layers_iou_div_mod(mypost_topn, model, generated_block_outputs)
                    # )

                    # to python lists
                    for ty in mypost_topn:
                        ty["top_n_probs"] = ty["top_n_probs"].detach().cpu().tolist()
                        ty["top_n_indices"] = ty["top_n_indices"].detach().cpu().tolist()
                    mypost_topk["top_k_probs"] = mypost_topk["top_k_probs"].detach().cpu().tolist()
                    mypost_topk["top_k_indices"] = (
                        mypost_topk["top_k_indices"].detach().cpu().tolist()
                    )

                    p["post_output_proba_topn"].append(mypost_topn)
                    p["post_output_proba_topk"].append(mypost_topk)

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

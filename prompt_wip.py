import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import gc
import os

# TODO: sample prompt (table length) + trim candidates to 10
# TODO: command-line args
# TODO: save params with the run
# TODO: logit lens (KL + IOU) for each layer of each token
# TODO: normalizzazione entropia
# TODO: check kuhn https://arxiv.org/pdf/2302.0966 https://arxiv.org/pdf/2307.10236

if 'HF_TOKEN' in os.environ: access_token = os.environ['HF_TOKEN']

# run params
temperature=1.0
top_p=0.9
max_new_tokens=128
use_cache=True
device = "mps"
# dict size

model_name = "osunlp/TableLlama"

# load inputs
file_path = "turl_test_2k_prompts_50.jsonl"
device = torch.device(device)

with open(file_path, "r", encoding="utf-8") as f:
    prompts = [json.loads(line) for line in f]

config = transformers.AutoConfig.from_pretrained(model_name)
orig_ctx_len = getattr(config, "max_position_embeddings", None)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.resize_token_embeddings(32001)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=orig_ctx_len, padding_side="left", use_fast=False)
model.eval()

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

def generate_prompt(instruction, question, input_seg=None):
    question += " Answer with just a candidate, selected from the provided referent entity candidates list, and nothing else. The selected candidate must be reported verbatim from the list provided as input. Each candidate in the list is enclosed between < and > and reports [DESC] and [TYPE] information."
    if input_seg:
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

def flip():
    gc.collect()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return

# get top-k (maximum probs)
def get_topk_dict(logits, k=10):
    probabilities = torch.softmax(logits, dim=-1)
    top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)
    top_k_probs = torch.gather(probabilities, dim=-1, index=top_k_indices)
    # return {'top_k_logits': top_k_values, 'top_k_probs': top_k_probs, 'top_k_indices': top_k_indices}
    return {'top_k_probs': top_k_probs, 'top_k_indices': top_k_indices}

# get the .99 probability mass (logits, probs and indices)
def get_topn_dict(logits, threshold=0.9):
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
        batched_results.append({
            #'top_n_logits': top_n_values[0, i, :n].to(torch.float32).cpu().numpy().tolist(),
            'top_n_probs': top_n_probs[0, i, :n].to(torch.float32).cpu().numpy().tolist(),
            'top_n_indices': top_n_indices[0, i, :n].to(torch.float32).cpu().numpy().tolist(),
        })

    return batched_results

import scipy.stats
def compute_entropy_scipy(logits):
    probabilities = torch.softmax(logits, dim=-1).cpu().float().numpy()
    entropy_values = [scipy.stats.entropy(row) for row in probabilities[0]]
    return entropy_values

outlist = []
for p in tqdm(prompts[:100]):
    try:
        prompt = generate_prompt(p["instruction"], p["question"], p["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            pre_output = model(**inputs, use_cache=use_cache)
        pre_output = pre_output.logits.cpu().detach()

        # top-n + top-k
        p["pre_output_proba_topn"] = get_topn_dict(pre_output)
        p["pre_output_proba_topk"] = get_topk_dict(pre_output)
        p["pre_output_true_entropies"] = compute_entropy_scipy(pre_output)

        # cleanup
        del pre_output
        flip()

        post_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=use_cache
            )

        # sequence
        post_output_sequences = post_output.sequences.cpu().detach().numpy().tolist()
        p["post_output_sequences"] = post_output_sequences

        post_output_scores = [pp.cpu().detach() for pp in post_output.scores]
        post_output_scores = torch.stack(post_output_scores, dim=1)
        
        # top-n + top-k
        p["post_output_proba_topn"] = get_topn_dict(post_output_scores)
        p["post_output_proba_topk"] = get_topk_dict(post_output_scores)
        p["post_output_true_entropies"] = compute_entropy_scipy(post_output_scores)

        # cleanup
        del post_output
        flip()

        outlist.append(p)

    except Exception as e:
        print("EXCEPTION!")
        import traceback
        print(traceback.format_exc())
        continue


import pickle
import random

myrand = str(random.randint(0, 2**32))
with open( model_name.split('/')[-1] + '.' + myrand + '.pickle', 'wb') as handle:
    pickle.dump(outlist, handle, protocol=pickle.HIGHEST_PROTOCOL)

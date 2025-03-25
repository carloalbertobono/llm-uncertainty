import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import copy
import json
from tqdm import tqdm
import gc
import os
import time

# TODO: command-line args
# TODO: save params with the run

if 'HF_TOKEN' in os.environ: access_token = os.environ['HF_TOKEN']

# run params
temperature=1.0
top_p=0.9
max_new_tokens=128
# use_cache=True
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
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
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

def get_layers_kl_div_mod(pre_output, model, precomp=None):
    softmaxed_log = torch.log_softmax(pre_output, dim=-1)

    kls = []
    for k, l in enumerate(model.model.layers):
        # print(softmaxed_log.shape, l.shape)
        
        # recover normalized from precomputed (if generated / hooks), or just read from network state
        if precomp: 
            mynorm = precomp[k]
        else:
            mynorm = l.mynorm
        
        l_ = l.unembed_matrix(mynorm)
        softmaxed_l = torch.log_softmax(l_, dim=-1).cpu()
        p = softmaxed_l[0]
        q = softmaxed_log[0]
        mykl = torch.nn.functional.kl_div(p, q, reduce=False, log_target=True).sum(dim=1)
        # print(p.sum(), q.sum(), mykl.shape)
        kls.append(mykl.detach().numpy())
    return kls

def jaccard_similarity(list1, list2):
    intersection = len(set(list1) & set(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return float(intersection) / union if union != 0 else 0.0

def get_layers_iou_div_mod(pre_output_proba_topk, model, precomp=None):
    
    ious = []
    
    # for each layer
    for k, l in enumerate(model.model.layers):
        
        # print(k)
        # recover normalized from precomputed (if generated / hooks), or just read from network state
        if precomp: 
            mynorm = precomp[k]
        else:
            mynorm = l.mynorm
        
        l_ = l.unembed_matrix(mynorm).detach()
        layer_topk = get_topn_dict(l_)
        
        # for each token
        layer_ious = []
        for a,b in zip(pre_output_proba_topk, layer_topk):
            a = a['top_n_indices']
            b = b['top_n_indices']   
            layer_ious.append(jaccard_similarity(a,b))
        ious.append(layer_ious)
            
    return ious

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.mynorm = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.mynorm = self.norm(output[0])
        # self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        return output

for i, layer in enumerate(model.model.layers):
    model.model.layers[i] = BlockOutputWrapper(layer, model.lm_head, model.model.norm)

# process only selected pickles
import pickle
with open('selected_pids.688.pickle', 'rb') as handle:
    selected_pids = pickle.load(handle)

NREP = 10 # number of multiple generate runs
MAXTOK = 2**12 # maximum sequence length

outlist = []

for pid, p in enumerate(tqdm(prompts)):
    try:
        start = time.time()

        # ( ! ) subsetting data
        if pid not in selected_pids: continue # focus on prompts where answer can be wrong

        prompt = generate_prompt(p["instruction"], p["question"], p["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # dirty trick for caching
        prompt_ = prompt[:-1]
        inputs_ = tokenizer(prompt_, return_tensors="pt").to(device)

        # ( ! )
        if inputs.input_ids.shape[1] > MAXTOK: 
            print("SKIPPED")
            continue # roughly 75% of data kept

        prompt_cache = StaticCache(config=model.config, max_batch_size=1, 
                           max_cache_len = MAXTOK, 
                           device='mps', dtype=torch.bfloat16)
        
        with torch.no_grad():
            # pre_output = model(**inputs, use_cache=use_cache)
            pre_output = model(**inputs_, past_key_values=prompt_cache, use_cache=False)
            prompt_cache = pre_output.past_key_values

        pre_output = pre_output.logits.detach().cpu()

        end = time.time()
        p['elapsed_pre'] = end - start

        # top-n + top-k
        p["pre_output_proba_topn"] = get_topn_dict(pre_output)
        p["pre_output_proba_topk"] = get_topk_dict(pre_output)
        p["pre_output_true_entropies"] = compute_entropy_scipy(pre_output)

        # logit lens KL/IOU
        # p["pre_output_layers_kl"] = get_layers_kl_div_mod(pre_output, model)
        # p["pre_output_layers_iou"] = get_layers_iou_div_mod(p["pre_output_proba_topn"], model)

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

            generated_block_outputs = {i: [] for i in range(len(model.model.layers))}

            def hook_fn(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple) and hasattr(module, "mynorm"):
                        clo = module.mynorm.clone().detach()
                        if clo.shape[1] > 1: 
                            shpbf = clo.shape
                            clo = clo[:,-1:,:] # first token returns all prompt tokens, patch last
                            # print(shpbf, clo.shape)
                        generated_block_outputs[layer_idx].append(clo)
                return hook
                
            # register hooks
            hooks = []

            for i, layer in enumerate(model.model.layers):
                hook = layer.register_forward_hook(hook_fn(i))
                hooks.append(hook)

            post_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                output_scores=True,
                return_dict_in_generate=True,
                output_logits=True,
                past_key_values=copy.deepcopy(prompt_cache),
                use_cache=True 
            )

            # remove hooks
            for hook in hooks:
                hook.remove()

            # accumulate norms at each generated token
            for i in generated_block_outputs:
                generated_block_outputs[i] = torch.stack(generated_block_outputs[i]).squeeze(1).permute(1, 0, 2)
            generated_block_outputs = list(generated_block_outputs.values())

            # sequence
            post_output_sequences = post_output.sequences.detach().cpu().numpy().tolist()
            p["post_output_sequences"].append(post_output_sequences)

            post_output_scores = [pp.detach().cpu() for pp in post_output.logits]
            post_output_scores = torch.stack(post_output_scores, dim=1)
            
            # top-n + top-k
            mypost_topn = get_topn_dict(post_output_scores)
            p["post_output_proba_topn"].append(mypost_topn)
            p["post_output_proba_topk"].append(get_topk_dict(post_output_scores))
            p["post_output_true_entropies"].append(compute_entropy_scipy(post_output_scores))

            # logit lens KL/IOU
            p["post_output_layers_kl"].append(get_layers_kl_div_mod(post_output_scores, model, generated_block_outputs))
            p["post_output_layers_iou"].append(get_layers_iou_div_mod(mypost_topn, model, generated_block_outputs))

            # transition scores
            transition_scores_s = model.compute_transition_scores(post_output.sequences, post_output.scores, normalize_logits=True)
            log_likelihoods_s = [score.item() for score in transition_scores_s[0]]
            p["transition_scores_s"].append(log_likelihoods_s)
            transition_scores_l = model.compute_transition_scores(post_output.sequences, post_output.logits, normalize_logits=True)
            log_likelihoods_l = [score.item() for score in transition_scores_l[0]]
            p["transition_scores_l"].append(log_likelihoods_l)

            # cleanup
            del post_output
            
        flip()

        # total processing time
        end = time.time()
        p['elapsed'] = end - start
        p['pid'] = pid

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

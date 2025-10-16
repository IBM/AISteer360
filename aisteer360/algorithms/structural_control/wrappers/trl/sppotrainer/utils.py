"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 The SPPO Authors and contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.

This file includes adapted portions of code from https://github.com/uclaml/SPPO (Apache License 2.0).
Modifications by IBM Research, 2025: refactoring, integration, bug fixes, and comments.
"""


# from https://github.com/uclaml/SPPO/blob/main/scripts/generate.py
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset


def set_seed(seed: int = 5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_template(text: str, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# from https://github.com/uclaml/SPPO/blob/main/scripts/rank.py
import llm_blender


def ranking(sppo_temp_dir: str, iter_num: int, prompts, candidates):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    ranks = blender.rank(prompts, candidates, return_scores=True, batch_size=1)
    os.makedirs(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}", exist_ok=True)
    np.save(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}/ranking.npy", ranks)


# from https://github.com/uclaml/SPPO/blob/main/scripts/compute_prob.py
def from_ranks(data: Dataset, pairs: int, sppo_temp_dir: str, iter_num: int):
    scores = np.load(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}/ranking.npy")
    scores = list(scores)

    probs = []
    rm_scores = []
    for score in scores:
        prb = np.zeros((pairs, pairs))
        for i in range(pairs):
            for j in range(pairs):
                prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
        probs.append(prb.tolist())
        rm_scores.append(score)

    os.makedirs(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}", exist_ok=True)
    with open(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/probabilities.json", "w") as f:
        json.dump(probs, f)

    df = data.to_pandas()
    for i in range(pairs):
        with open(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/responses_{i}.json") as f:
            responses = json.load(f)
        fmt = [
            [
                {"content": data[j]["prompt"], "role": "user"},
                {"content": responses[j], "role": "assistant"},
            ]
            for j in range(len(data))
        ]
        df[f"generate_{i}"] = fmt

    if pairs < 5:
        cols_to_delete = [f"generate_{ind}" for ind in range(pairs, 5) if f"generate_{ind}" in df.columns]
        if cols_to_delete:
            df.drop(cols_to_delete, axis=1, inplace=True)

    df["probability"] = probs
    df["rm_scores"] = rm_scores
    df.to_parquet(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/train.parquet")


def prepare_score(pairs: int, sppo_temp_dir: str, iter_num: int):
    train = Dataset.from_parquet(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/train.parquet")
    train = pd.DataFrame(train)

    metrics = train["rm_scores"].apply(lambda x: np.array(x[-pairs:]))
    metrics_prob = train["probability"].apply(lambda x: np.stack(x).sum(axis=1))
    maxmin = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    columns = [f"generate_{ind}" for ind in range(pairs)] + ["probability"]
    train_ordered = train[columns]

    chosen = [train_ordered.iloc[i, maxmin[i][0]] for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin[i][1]] for i in range(len(train_ordered))]

    chosen_probs = [train_ordered["probability"].iloc[i][maxmin[i][0]][maxmin[i][1]] for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob[i][maxmin[i][0]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob[i][maxmin[i][1]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]

    out_path = f"{sppo_temp_dir}/synthetic_data_SPPO-Iter{iter_num}_score"
    os.makedirs(out_path, exist_ok=True)

    train_new = pd.DataFrame(
        {
            "chosen": chosen,
            "rejected": rejected,
            "chosen_probs": chosen_probs,
            "chosen_probs_win": chosen_probs_win,
            "chosen_probs_lose": chosen_probs_lose,
        }
    )
    train_new.to_parquet(f"{out_path}/train.parquet", index=False)

    test = train_new.sample(n=max(1, int(0.1 * len(train_new))))
    test.to_parquet(f"{out_path}/test.parquet", index=False)

    return out_path


def apply_chat_template(example, tokenizer, skip_system_message: bool):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        if not skip_system_message:
            prompt_messages = example["chosen"][:-1]
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False, add_generation_prompt=True
            )[len(example["text_prompt"]):]
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False, add_generation_prompt=True
            )[len(example["text_prompt"]):]
    else:
        raise ValueError(
            f"Could not format example as dialogue for `sppo` task! "
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def process_dataset(raw_dataset: Dataset, tokenizer):
    column_names = [x for x in list(raw_dataset.features) if x not in ["chosen_probs", "chosen_probs_win", "chosen_probs_lose"]]

    raw_dataset = raw_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "skip_system_message": True},
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    raw_dataset = raw_dataset.rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )
    return raw_dataset


def prepare_dataset_from_prompts(
    llm,
    tokenizer,
    data: Dataset,
    sppo_temp_dir: str,
    iter_num: int = 1,
    maxlen: int = 2048,
    num_prompts: int = 5,
    gen_max_new_tokens: int = 128,
    ranking_batch_size: int = 8,
    limit_num_examples: int | None = None,
):
    """
    Prepare a processed training dataset for SPPO:
      1) Generate K responses per prompt with the current model (left padding, short outputs).
      2) Rank responses with PairRM (batched).
      3) Convert to (prompt, chosen, rejected) via chat templates.

    Notes:
      - `maxlen` is used for input prompt truncation.
      - `gen_max_new_tokens` limits output length (prevents hours-long runs).
      - padding_side='left' set temporarily for decoder-only models.
    """
    device = next(llm.parameters()).device

    # optionally cap the dataset size for faster iterations
    if limit_num_examples is not None and limit_num_examples < len(data):
        data = data.select(range(limit_num_examples))

    iter_gen_dir = f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}"
    Path(iter_gen_dir).mkdir(parents=True, exist_ok=True)

    # build prompt strings with add_generation_prompt=True
    prompts = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]

    # force left padding just for generation (restore afterwards)
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for k in range(num_prompts):
            set_seed(k * 50)

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=maxlen,  # truncate input prompts
            )
            enc = enc.to(device) if hasattr(enc, "to") else {kk: (vv.to(device) if torch.is_tensor(vv) else vv)
                                                             for kk, vv in enc.items()}

            llm.eval()
            with torch.inference_mode():
                generated_ids = llm.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=gen_max_new_tokens,  # short outputs
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )

            input_length = enc["input_ids"].shape[1]
            generated_text = tokenizer.batch_decode(
                generated_ids[:, input_length:], skip_special_tokens=True
            )

            with open(f"{iter_gen_dir}/responses_{k}.json", "w") as f:
                json.dump(generated_text, f)
    finally:
        tokenizer.padding_side = original_padding_side

    # rank with PairRM (batched instead of batch_size=1)
    all_generated = []
    for k in range(num_prompts):
        with open(f"{iter_gen_dir}/responses_{k}.json") as f:
            all_generated.append(json.load(f))

    candidates_texts = list(zip(*all_generated))
    assert len(data) == len(candidates_texts)
    os.makedirs(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}", exist_ok=True)

    # allow a larger batch size for throughput
    blender = llm_blender.Blender()
    try:
        # some versions accept device/torch_dtype; ignore if unsupported
        blender.loadranker("llm-blender/PairRM")
        ranks = blender.rank(prompts, candidates_texts, return_scores=True,
                             batch_size=max(1, ranking_batch_size))
    except TypeError:
        # fallback to the original signature
        blender.loadranker("llm-blender/PairRM")
        ranks = blender.rank(prompts, candidates_texts, return_scores=True, batch_size=1)

    np.save(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}/ranking.npy", ranks)

    # compute probabilities and build pairwise data
    from_ranks(data, num_prompts, sppo_temp_dir, iter_num)
    out_path = prepare_score(num_prompts, sppo_temp_dir, iter_num)

    train = Dataset.from_parquet(f"{out_path}/train.parquet")
    processed_train = process_dataset(train, tokenizer)
    return processed_train


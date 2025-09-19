# from https://github.com/uclaml/SPPO/blob/main/scripts/generate.py
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_template(text, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]


# from https://github.com/uclaml/SPPO/blob/main/scripts/rank.py
import llm_blender


def ranking(sppo_temp_dir, iter_num, prompts, candidates):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    ranks = blender.rank(prompts, candidates, return_scores=True, batch_size=1)
    np.save(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}/ranking.npy", ranks)



# from https://github.com/uclaml/SPPO/blob/main/scripts/compute_prob.py

# --output_dir $OUTDIR --pairs $PAIRS --prompts $PROMPTS

def from_ranks(data, pairs, sppo_temp_dir, iter_num):
    scores = np.load(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}/ranking.npy")  # ranking_fp == "./ranking.npy")
    scores = list(scores)

    probs = []
    rm_scores = []
    for idx, score in enumerate(scores):
        prb = np.zeros((pairs, pairs))
        for i in range(pairs):
            for j in range(pairs):
                prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
        prb = prb.tolist()
        probs.append(prb)
        rm_scores.append(score)

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
    if pairs < 5: #original implementation assumes pairs == 5
        #remove extra generate_x columns if they exist
        cols_to_delete = []
        for ind in range(pairs, 5):
            if f"generate_{ind}" in df.columns:
                cols_to_delete.append(f"generate_{ind}")
        if (len(cols_to_delete) > 0):
            df.drop(cols_to_delete, axis=1, inplace=True)
    df["probability"] = probs
    df["rm_scores"] = rm_scores
    df.to_parquet(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/train.parquet")


def prepare_score(pairs, sppo_temp_dir, iter_num):
    # Load dataset and convert to DataFrame
    train = Dataset.from_parquet(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/train.parquet")
    train = pd.DataFrame(train)

    # Calculate metrics and probabilities
    metrics = train['rm_scores'].apply(lambda x: np.array(x[-pairs:]))
    metrics_prob = train['probability'].apply(lambda x: np.stack(x).sum(axis=1))
    maxmin = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    # Reorganize the DataFrame for easy access
    cols = []
    for ind in range(pairs):
        cols.append(f"generate_{ind}")
    cols.append('probability')
    train_ordered = train[cols]

    # Determine chosen and rejected items based on maxmin indices
    chosen = [train_ordered.iloc[i, maxmin[i][0]] for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin[i][1]] for i in range(len(train_ordered))]

    # Calculate probabilities for chosen and rejected items
    chosen_probs = [train_ordered['probability'].iloc[i][maxmin[i][0]][maxmin[i][1]] for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob[i][maxmin[i][0]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob[i][maxmin[i][1]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]

    # Create a new DataFrame with the results
    train_new = pd.DataFrame({
        'chosen': chosen,
        'rejected': rejected,
        'chosen_probs': chosen_probs,
        'chosen_probs_win': chosen_probs_win,
        'chosen_probs_lose': chosen_probs_lose
    })

    # Determine output directory
    #output_dir = '-'.join(output_dir.split('-')[1:])
    OUTPATH = f'{sppo_temp_dir}/synthetic_data_SPPO-Iter{iter_num}_score'
    os.makedirs(OUTPATH, exist_ok=True)

    # Save train and test datasets to parquet files
    train_new.to_parquet(f'{OUTPATH}/train.parquet', index=False)
    print(f"Saved file to {OUTPATH}/train.parquet")

    # Temporary solution to make the code run, cannot use for test/evaluation purpose
    test = train_new.sample(n=int(0.1*len(train_new)))
    test.to_parquet(f'{OUTPATH}/test.parquet', index=False)
    print(f"Saved file to {OUTPATH}/test.parquet")

    return OUTPATH


def apply_chat_template(
    example,
    tokenizer,
    skip_system_message,
):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        prompt_messages = example["chosen"][:-1]
        # Prepend a system message if the first message is not a system message
        if not skip_system_message:
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False, add_generate_prompt=True
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False, add_generate_prompt=True
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generate_prompt=True
            )
        else:
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generate_prompt=True
            )
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False, add_generate_prompt=True
            )[len(example["text_prompt"]) :]
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False, add_generate_prompt=True
            )[len(example["text_prompt"]) :]
    else:
        raise ValueError(
            f"Could not format example as dialogue for `sppo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example

def process_dataset(raw_dataset, tokenizer):
    column_names = list(raw_dataset.features)
    column_names = [x for x in column_names if x not in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']]

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

def prepare_dataset_from_prompts(llm, tokenizer, data, sppo_temp_dir, iter_num=1, maxlen = 2048, num_prompts=5):

    # Step 1: https://github.com/uclaml/SPPO/blob/main/scripts/generate.sh
    # Step 1(a): Generate data - https://github.com/uclaml/SPPO/blob/main/scripts/generate.py
    Path(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}").mkdir(parents=True, exist_ok=True)
    prompts = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]

    for p in range(num_prompts):
        set_seed(p * 50)
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        generated_ids = llm.generate(**enc,  do_sample=True, temperature=0.7,top_p=0.9,max_new_tokens=maxlen)

        generated_text = tokenizer.batch_decode(generated_ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True) #[0]  #

        with open(f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/responses_{p}.json", "w") as f:
            json.dump(generated_text, f)


    # Step 1(b): Rank data - https://github.com/uclaml/SPPO/blob/main/scripts/rank.py
    all_generated = []

    for i in range(num_prompts):
        file_path = f"{sppo_temp_dir}/generated/SPPO-Iter{iter_num}/responses_{i}.json"
        with open(file_path) as f:
            gen = json.load(f)
            all_generated.append(gen)

    candidates_texts = list(zip(*all_generated))
    assert len(data) == len(candidates_texts)

    os.makedirs(f"{sppo_temp_dir}/ranking/SPPO-Iter{iter_num}", exist_ok=True)

    ranking(sppo_temp_dir, iter_num, prompts, candidates_texts)

    # Step 1(c): Compute probs - https://github.com/uclaml/SPPO/blob/main/scripts/compute_prob.py
    from_ranks(data, num_prompts, sppo_temp_dir, iter_num)
    out_path = prepare_score(num_prompts, sppo_temp_dir, iter_num)
    print('probs calculated')

    # Step 2: https://github.com/uclaml/SPPO/blob/main/scripts/pipeline.sh
    train = Dataset.from_parquet(f"{out_path}/train.parquet")
    processed_train = process_dataset(train, tokenizer)
    return processed_train

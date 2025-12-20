import json

from datasets import Dataset
from peft import PeftType

from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.input_control.few_shot import FewShot
from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer import DPO
from aisteer360.evaluation.benchmark import Benchmark
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_accuracy import (
    MCQAAccuracy,
)
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_positional_bias import (
    MCQAPositionalBias,
)
from aisteer360.evaluation.use_cases.commonsense_mcqa.use_case import CommonsenseMCQA

commonsense_mcqa = CommonsenseMCQA(
    evaluation_data="../data/evaluation_qa.jsonl",
    evaluation_metrics=[
        MCQAAccuracy(),
        MCQAPositionalBias(),
    ],
    num_shuffling_runs=20,
    num_samples=50  # optional
)


steering_data_path = "../data/steer_qa.jsonl"
with open(steering_data_path, "r") as f:
    steering_data = [json.loads(line) for line in f]


# few-shot control spec
positive_pool = []
negative_pool = []
for row in steering_data:
    positive_pool.append({
        "question": row["question"],
        "answer": row["answer_chosen"]
    })
    negative_pool.append({
        "question": row["question"],
        "answer": row["answer_rejected"]
    })

few_shot_spec = ControlSpec(
    control_cls=FewShot,
    params={
        "selector_name": "random",
        "positive_example_pool": positive_pool,
        "negative_example_pool": negative_pool
    },
    vars={
        "k_positive": [5, 10, 20],
        "k_negative": [5, 10, 20]
    },
    name="few_shot",
)


# DPO control spec
train_examples = []
for row in steering_data:
    train_examples.append({
        "prompt": row['question'],
        "chosen": row['answer_chosen'],
        "rejected": row['answer_rejected']
    })
train_ds = Dataset.from_list(train_examples)

dpo_lora_spec = ControlSpec(
    control_cls=DPO,
    params={
        "train_dataset": train_ds,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 2,
        "learning_rate": 1e-6,
        "loss_type": "sigmoid",
        "max_length": 1024,
        "max_prompt_length": 512,
        "disable_dropout": True,
        "logging_steps": 100,
        "save_strategy": "no",
        "report_to": "none",
        "seed": 123,
        "use_peft": True,
        "peft_type": PeftType.LORA,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "adapter_name": "dpo",
        "merge_lora_after_train": False,
        "output_dir": "runs/bench_dpo/",
    },
    vars={
        "beta": [0.05, 0.1],
        "r": [8, 16]
    },
    name="dpo",
)


# Run benchmark
bench = Benchmark(
    use_case=commonsense_mcqa,
    base_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    steering_pipelines={
        "baseline": [],
        "few_shot": [few_shot_spec],
        "dpo_lora": [dpo_lora_spec],
    },
    gen_kwargs={
        "max_new_tokens": 300,
        "do_sample": True,
        "temperature": 0.7
    },
    device_map="auto",
    num_trials=5
)

profiles = bench.run()

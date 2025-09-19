"""
The TRL wrapper implements a variety of methods from Hugging Face's [TRL library](https://huggingface.co/docs/trl/index).

The current functionality spans the following methods:

- **SFT (Supervised Fine-Tuning)**: Standard supervised learning to fine-tune language models on demonstration data
- **DPO (Direct Preference Optimization)**: Trains models directly on preference data without requiring a separate reward model
- **APO (Anchored Preference Optimization)**: A variant of DPO that uses an anchor model to improve training stability and performance
- **SPPO (Self-Play Preference Optimization)**: Iterative preference optimization using self-generated synthetic data to reduce dependency on external preference datasets

For documentation information, please refer to the [TRL page](https://huggingface.co/docs/trl/index) and the [SPPO repository](https://github.com/uclaml/SPPO).

"""

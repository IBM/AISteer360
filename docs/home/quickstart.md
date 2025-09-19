# Quickstart

This guide will walk you through how to run a simple control in AISteer360.

!!! note
    AISteer360 runs the model inside your process. For efficient inference on more complex steering operations, please
    run the toolkit from a machine that has enough GPU memory for both the base checkpoint and the extra overhead your
    steering method/pipeline adds.

The first step in steering any model is to define how you want to steer, i.e., the control. For this guide, we will
implement a simple `FewShot` control in which we influence model behavior via few-shot examples. The desired target
behavior for this example is "conciseness".

The `FewShot` control requires specification of example pools to draw from. A set of positive example, i.e., in which
the model provided a concise answer, are defined as follows:

```python
positive_example_pool = [
    {"question": "What's the capital of France?", "answer": "Paris"},
    {"question": "How many miles is it to the moon?", "answer": "238,855"},
    {"question": "What's 15% of 200?", "answer": "30"},
    {"question": "What's the boiling point of water?", "answer": "100°C"},
    {"question": "How many days in a leap year?", "answer": "366"},
    {"question": "What's the speed of light?", "answer": "299,792,458 m/s"},
    {"question": "How many continents are there?", "answer": "7"},
    {"question": "What's the atomic number of gold?", "answer": "79"}
]
```

Similarly, we define the following negative examples where the model was not concise:
```python
negative_example_pool = [
    {"question": "What's the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "How many miles is it to the moon?", "answer": "The Moon is an average of 238,855 miles (384,400 kilometers) away from Earth."},
    {"question": "What's the boiling point of water?", "answer": "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at sea level."},
    {"question": "How many days in a leap year?", "answer": "A leap year contains 366 days, which is one day more than a regular year."},
    {"question": "What's the speed of light?", "answer": "The speed of light in vacuum is approximately 299,792,458 meters per second."},
    {"question": "What's 15% of 200?", "answer": "Fifteen percent of 200 can be calculated by multiplying 200 by 0.15, which gives 30."},
    {"question": "How many continents are there?", "answer": "There are seven continents on Earth: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America."},
    {"question": "What's the atomic number of gold?", "answer": "Gold has the atomic number 79 on the periodic table of elements."}
]
```

Using these pools, we define the `FewShot` control as follows:
```python
from aisteer360.algorithms.input_control.few_shot.control import FewShot

few_shot = FewShot(
    selector_name="random",
    positive_example_pool=positive_example_pool,
    negative_example_pool=negative_example_pool,
    k_positive=4,
    k_negative=4
)
```

We can then define a `SteeringPipeline` on a given base model using the above control:
```python
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
few_shot_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[few_shot],
    device_map="auto"
)
few_shot_pipeline.steer()
```

Inference can now be run on the steered pipeline as follows:
```python
prompt = "How many feet are in a mile?"
input_ids = few_shot_pipeline.tokenizer.encode(prompt, return_tensors="pt")

output = few_shot_pipeline.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.7,
    return_full_sequence=False
)

print(few_shot_pipeline.tokenizer.decode(output[0], skip_special_tokens=True))
```

And there you have it, a simple few-shot steering control. For more complex controls, as well as examples on how
controls can be compared on a given task, please see the [example notebooks](../notebooks/index.md).

import multiprocessing as mp

MODEL = ""


def main():
    from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
    from aisteer360.algorithms.state_control.act_add import ActAdd
    from aisteer360.algorithms.state_control.caa import CAA
    from aisteer360.algorithms.state_control.iti import ITI

    # add backend="vllm"
    pipe = SteeringPipeline(
        model_name_or_path=MODEL,
        controls=[ActAdd(positive_prompt="Love", negative_prompt="Hate", multiplier=5.0)],
        backend="vllm",
        vllm_kwargs={"gpu_memory_utilization": 0.9},
    )
    # pipe = SteeringPipeline(
    #     model_name_or_path=MODEL,
    #     controls=[CAA(                                                                                         
    #       data={  
    #           "positives": ["Love", "Kindness", "Joy", "Compassion"],                                         
    #           "negatives": ["Hate", "Cruelty", "Sadness", "Indifference"],                                    
    #       },                                                                                                 
    #       multiplier=5.0,                                                                                    
    #       token_scope="all",  # important for vLLM — "after_prompt" won't work                               
    #   )],
    #   backend="vllm",
    #     vllm_kwargs={"gpu_memory_utilization": 0.9},
    # )
    # pipe = SteeringPipeline(
    #     model_name_or_path=MODEL,
    #     controls=[ITI(                                                                                             
    #         data={                                                                                                 
    #             "positives": [                                                                                     
    #                 "The sky is blue.",
    #                 "Water boils at 100 degrees Celsius.",                                                         
    #                 "The Earth orbits the Sun.",                                                                   
    #                 "Humans need oxygen to breathe.",                                                              
    #             ],                                                                                                 
    #             "negatives": [                                                                                     
    #                 "The sky is green.",
    #                 "Water boils at 50 degrees Celsius.",                                                          
    #                 "The Sun orbits the Earth.",                                                                   
    #             ],
    #         },                                                                                                     
    #         alpha=15.0, 
    #         num_heads=48,                                                                                          
    #         token_scope="all",  # important for vLLM
    #     )],
    #     backend="vllm",
    #     vllm_kwargs={"gpu_memory_utilization": 0.9},
    # )
    pipe.steer()

    # Compare steered vs baseline
    ids = pipe.tokenizer(["Tell me a story."], return_tensors="pt")["input_ids"]
    steered = pipe.generate_text(ids, use_hook=True)
    pipe._vllm_engine.llm_engine.reset_prefix_cache()
    baseline = pipe.generate_text(ids, use_hook=False)

    print("\n=== Steered ===")
    print(steered)
    print("\n=== Baseline ===")
    print(baseline)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

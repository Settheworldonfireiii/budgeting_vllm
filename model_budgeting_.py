from vllm import LLM, SamplingParams
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from datasets import load_dataset
import argparse
import json




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',  help="model name, including vendor", type= str)
    parser.add_argument('--min_tokens',  help="minimal amount of tokens that the model must think for during each generation, whether original one or extended", type= int)
    parser.add_argument('--max_tokens',  help="token threshold, or maximal amount of tokens that we allow before we interrupt the model and force it to give the answer", type= int)
    parser.add_argument('--max_final_tokens',  help="maximum amount of tokens the model is allowed to output after its thinking was interrupted and it is forced to give a final answer", type= int)
    parser.add_argument('--temperature',  help="generation temperature", type= float)
    parser.add_argument('--num_ignore',  help="how many times to ignore end-of-thinking token", type= int)
    parser.add_argument('--dataset',  help="dataset name; currently supports only VanWang/NuminaMath-CoT_O1_Qwq and Open-COT-Data/COT-Dataset-Math", nargs='?', const = "VanWang/NuminaMath-CoT_O1_Qwq", default = "VanWang/NuminaMath-CoT_O1_Qwq", type= str)
    parser.add_argument('--ngpus',  help="how many gpus are you using? to parallelize", nargs = '?', const = 1, default = 1, type= int)

    args=parser.parse_args()
    # Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
    # Decide how often to ignore end-of-thinking token
    NUM_IGNORE = args.num_ignore



    model = LLM(
        args.model_name, # s1 originally gets this prompt wrong but with budget forcing it fixes it
        tensor_parallel_size=args.ngpus,
        )

    tok = AutoTokenizer.from_pretrained(
        args.model_name
    )
    

    stop_token_ids = tok("<|im_end|>")["input_ids"]
    ds = load_dataset(args.dataset)




    prompts = ds['train']
    data = []
    for i, p in enumerate(prompts):
        if args.dataset =="Open-COT-Data/COT-Dataset-Math":
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p['instruction'] + p['input'] + "<|im_end|>\n<|im_start|>assistant\n"
        elif args.dataset == "VanWang/NuminaMath-CoT_O1_Qwq":
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p['prompt'] + "<|im_end|>\n<|im_start|>assistant\n"
        elif args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            prompt = "<|im_start|>system\n " + p['system']  + "<|im_end|>\n<|im_start|>user\n" + p['conversations'][0]['value'] + "<|im_end|>\n<|im_start|>assistant\n"

        original_prompt = prompt
        orpleen  = len(original_prompt)
        stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=args.temperature,
        )
        prompt += "<|im_start|>think"
        o = model.generate(
            prompt,
            sampling_params=sampling_params
        )
        ignore_str = "Wait, think again"
        max_tokens_thinking_tmp = args.max_tokens
        # Num of times to skip stop token
        for k in range(NUM_IGNORE):
            print("ignore ", k)
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            prompt += o[0].outputs[0].text + ignore_str
            sampling_params = SamplingParams(
                max_tokens= max(max_tokens_thinking_tmp, 400),
                min_tokens = min(args.min_tokens, max(max_tokens_thinking_tmp, 400)),
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=args.temperature,
            )
            o = model.generate(
                prompt,
                sampling_params=sampling_params
            )
        ### Final answer ###
        prompt += o[0].outputs[0].text # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
        stop_token_ids = tok("<|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=args.max_final_tokens,
            min_tokens=min(args.min_tokens, args.max_final_tokens),
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=args.temperature,
        )
        o = model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        print("With budget forcing:") # You will see that after the "Wait" in the reasoning trace it fixes its answer
        print(prompt + o[0].outputs[0].text)
        if args.dataset == "VanWang/NuminaMath-CoT_O1_Qwq":
            entry = { 
                     'number' : i,
                     'prompt' : original_prompt,
                     'accepted' : p['chosen'],
                     'rejected' : prompt[orpleen+31:] + o[0].outputs[0].text
                     }
        elif args.dataset == "Open-COT-Data/COT-Dataset-Math":
            entry = {                   
                     'number' : i,
                     'prompt' : original_prompt,
                     'accepted' : p['output'],
                     'rejected' : prompt[orpleen+31:] + o[0].outputs[0].text
                     }
        elif args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            entry = {
                     'number' : i,
                     'prompt' : original_prompt,
                     'accepted' : p['conversations'][1]['value'],
                     'rejected' : prompt[orpleen+31:] + o[0].outputs[0].text
                     }

        data.append(entry)
        
        with open(f"{args.dataset[:11]}.json", "w") as outfile:
            json.dump(data, outfile, indent=4)

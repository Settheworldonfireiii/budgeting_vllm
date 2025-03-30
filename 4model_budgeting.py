from vllm import LLM, SamplingParams
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from datasets import load_dataset
import argparse
import json

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',  help="model name, including vendor", type= str)
    parser.add_argument('--min_tokens',  help="minimal amount of tokens that the model must think for during each generation, whether original one or extended", nargs = '?', const = 0, default = 0, type= int)
    parser.add_argument('--max_tokens',  help="token threshold, or maximal amount of tokens that we allow before we interrupt the model and force it to give the answer", type= int)
    parser.add_argument('--max_final_tokens',  help="maximum amount of tokens the model is allowed to output after its thinking was interrupted and it is forced to give a final answer", nargs = '?', const = 0, default = 0, type= int)
    parser.add_argument('--temperature',  help="generation temperature", type= float)
    parser.add_argument('--num_ignore',  help="how many times to ignore end-of-thinking token", nargs = '?', const = 0, default = 0,  type= int)
    parser.add_argument('--dataset',  help="dataset name; currently supports only VanWang/NuminaMath-CoT_O1_Qwq and Open-COT-Data/COT-Dataset-Math", nargs='?', const = "VanWang/NuminaMath-CoT_O1_Qwq", default = "VanWang/NuminaMath-CoT_O1_Qwq", type= str)
    parser.add_argument('--ngpus',  help="how many gpus are you using? to parallelize", nargs = '?', const = 1, default = 1, type= int)
    parser.add_argument('--topp',  help="top-p", nargs = '?', const = 0.01, default = 0.01, type= float)
    parser.add_argument('--mode',  help="do we want to keep generating until BOTH num_ignore and max_tokens expire, or until ONE OF THEM expires", nargs = '?', const = "lax", default = "lax", type= str)
    parser.add_argument('--custom_prompt',  help="do we want some custom prompt before asking the math question itself", nargs = '?', const = "", default = "", type= str)
    parser.add_argument('--cp_arg_1',  help="custom prompt input argument e g text, or input, or insturction, specific to dataset", nargs = '?', const = "", default = "", type= str)
    parser.add_argument('--cp_arg_2',  help="custom prompt input argument e g text, or input, or insturction, specific to dataset", nargs = '?', const = "", default = "", type= str)
    parser.add_argument('--cp_arg_3',  help="custom prompt input argument e g text, or input, or insturction, specific to dataset", nargs = '?', const = "", default = "", type= str)
    parser.add_argument('--custom_final_answer_prompt',  help="if we generate the answer separately, do we also generate it with some custom prompt?", nargs = '?', const = "\n Final Answer: ", default = "\n Final Answer", type= str)
    parser.add_argument('--ignore_string',  help="Some custom ignore string if Wait, think again is not OK for you", nargs = '?', const = ". Wait, think again", default = ". Wait, think again", type= str)
    

    args=parser.parse_args()
    # Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
    # Decide how often to ignore end-of-thinking token
    NUM_IGNORE = args.num_ignore



    model = LLM(
        args.model_name, # s1 originally gets this prompt wrong but with budget forcing it fixes it
        tensor_parallel_size=args.ngpus,)

    tok = AutoTokenizer.from_pretrained(
        args.model_name
    )
    

    stop_token_ids = tok("<|im_end|>")["input_ids"]
    ds = load_dataset(args.dataset)




    prompts = ds['train']
    data = []
    for i, p in enumerate(prompts):
        if args.custom_prompt:
            prompt += args.custom_prompt
            if args.cp_arg_1:
                prompt += p[args.cp_arg_1]
                if args.cp_arg_2:
                    prompt += p[args.cp_arg_2]
                    if args.cp_arg_3:
                        prompt += p[args.cp_arg_3]
            prompt += "<|im_end|>\n<|im_start|>assistant\n"  
        else:
            if args.dataset =="Open-COT-Data/COT-Dataset-Math":
                prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p['instruction'] + p['input'] + "<|im_end|>\n<|im_start|>assistant\n"
            elif args.dataset == "VanWang/NuminaMath-CoT_O1_Qwq":
                prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p['prompt'] + "<|im_end|>\n<|im_start|>assistant\n"
            elif args.dataset == "bespokelabs/Bespoke-Stratos-17k":
                prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.  Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. <|im_end|>\n<|im_start|>user\n" + p['conversations'][0]['value'] +  "<|im_end|>\n<|im_start|>assistant\n"
            #prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p['conversations'][0]['value'] + "<|im_end|>\n<|im_start|>assistant\n"

        original_prompt = prompt
        orpleen  = len(original_prompt)
        stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
        """
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            stop_token_ids = tok("<|begin_of_thought|> <|end_of_thought|>")["input_ids"]
        else:
            stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
        """
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=args.temperature,
            top_p = args.topp,
        )
        """
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            prompt += "<|begin_of_thought|> "
        else:
            prompt += "<|im_start|>think"
        """
        prompt += "<|im_start|>think"
        pdb.set_trace()
        o = model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        pdb.set_trace() 
        max_tokens_thinking_tmp = args.min_tokens
        print("=====================FIRST==========================================")
        print(o[0].outputs[0].text)
        # Num of times to skip stop token
        if args.mode == "lax":          
            while ((NUM_IGNORE>0) and (max_tokens_thinking_tmp > 100)):
                print("ignore ", NUM_IGNORE)
                max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
                prompt += o[0].outputs[0].text + args.ignore_string
                sampling_params = SamplingParams(
                    max_tokens= max(max_tokens_thinking_tmp, 400),
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=args.temperature,
                    top_p = args.topp,
                )
                o = model.generate(
                    prompt,
                    sampling_params=sampling_params
                )
                print(f"===================== {NUM_IGNORE}==========================================")
                print(o[0].outputs[0].text)
                NUM_IGNORE -= 1
        else:
            while ((NUM_IGNORE>0) or (max_tokens_thinking_tmp > 100)):
                print("ignore ", NUM_IGNORE)
                max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
                prompt += o[0].outputs[0].text + args.ignore_string
                sampling_params = SamplingParams(
                    max_tokens= max(max_tokens_thinking_tmp, 400),
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=args.temperature,
                    top_p = args.topp,
                )
                o = model.generate(
                    prompt,
                    sampling_params=sampling_params
                )
                print(f"===================== {NUM_IGNORE}==========================================")
                print(o[0].outputs[0].text)
                NUM_IGNORE -= 1

        ### Final answer ###
        prompt += o[0].outputs[0].text + args.custom_final_answer_prompt # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
        
        
        """
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            prompt += "<|im_start|>system\n Now, give the final answer, based on various attempts, explorations, and reflections from the chain of thought you previously generated, systematically present the final solution that you deem correct. The solution should have a logical, accurate, concise expression style and detail necessary steps needed to reach the conclusion.<|im_end|>"
            stop_token_ids = tok("<|end_of_solution|>")["input_ids"]
        else: 
            stop_token_ids = tok("<|im_end|>")["input_ids"]
        """
        if args.max_final_tokens > 0:
            stop_token_ids = tok("<|im_end|>")["input_ids"]
            sampling_params = SamplingParams(
                max_tokens=args.max_final_tokens,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=args.temperature,
                top_p = args.topp,
            )
            banlen = len(o[0].outputs[0].text)
            o = model.generate(
                prompt,
                sampling_params=sampling_params,
            )
        print("With budget forcing:") # You will see that after the "Wait" in the reasoning trace it fixes its answer
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
                      
            print("==========================+ANSWER END+==========")
            print(o[0].outputs[0].text)
            print("=============AFTER ONE MORE TRY===========================")
            print(prompt[orpleen:] + o[0].outputs[0].text)
        else:
            print(prompt[orpleen+31:] + o[0].outputs[0].text)
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
                     'rejected' : prompt[orpleen:] + o[0].outputs[0].text
                     }

        data.append(entry)
        
        with open(f"{args.dataset[:11]}.json", "w") as outfile:
            json.dump(data, outfile, indent=4)

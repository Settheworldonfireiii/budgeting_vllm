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
    parser.add_argument('--topp',  help="top-p", nargs = '?', const = 0.01, default = 0.01, type= float)

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
            prompt = "<|im_start|>system\n " + p['system'] +"for example, <|im_start|>user\n" + "Return your final response within \\boxed{}. The operation $\\otimes$ is defined for all nonzero numbers by $a\\otimes b =\\frac{a^{2}}{b}$. Determine $[(1\\otimes 2)\\otimes 3]-[1\\otimes (2\\otimes 3)]$.\n$\\text{(A)}\\ -\\frac{2}{3}\\qquad\\text{(B)}\\ -\\frac{1}{4}\\qquad\\text{(C)}\\ 0\\qquad\\text{(D)}\\ \\frac{1}{4}\\qquad\\text{(E)}\\ \\frac{2}{3}$<|im_end|>\n<|im_start|>assistant \n <|begin_of_thought|>\n\nOkay, let me try to figure out this problem. So, we have this operation defined as a⊗b = a²/b. And we need to compute [(1⊗2)⊗3] - [1⊗(2⊗3)]. Then choose the correct answer from the options given. Alright, let's break it down step by step.\n\nFirst, I need to remember that the operation ⊗ is not associative, right? Because the problem is asking for the difference between two different groupings: (1⊗2)⊗3 and 1⊗(2⊗3). So, the order in which we perform the operations matters here. That's probably why there's a subtraction between them.\n\nLet me start by computing each part separately. Let's tackle the first part: (1⊗2)⊗3.\n\nStarting with the innermost operation, which is 1⊗2. According to the definition, a⊗b = a²/b. So here, a is 1 and b is 2. Plugging those in: 1² / 2 = 1/2. So, 1⊗2 equals 1/2.\n\nNow, we take that result and perform the next operation with 3. So, (1⊗2)⊗3 becomes (1/2)⊗3. Again, using the same definition: a is now 1/2 and b is 3. So, ( (1/2)² ) / 3 = (1/4) / 3 = 1/12. So, (1⊗2)⊗3 equals 1/12.\n\nAlright, that's the first part. Now let's compute the second part: 1⊗(2⊗3). Again, starting with the innermost operation, which is 2⊗3. Applying the definition: a is 2 and b is 3. So, 2² / 3 = 4/3. Therefore, 2⊗3 equals 4/3.\n\nNow, we need to compute 1⊗(4/3). Here, a is 1 and b is 4/3. Using the operation definition: 1² / (4/3) = 1 / (4/3) = 3/4. So, 1⊗(2⊗3) equals 3/4.\n\nNow, the problem asks for the difference between the two results: [(1⊗2)⊗3] - [1⊗(2⊗3)] = (1/12) - (3/4). To subtract these fractions, they need a common denominator. The denominators are 12 and 4, so 12 is the common denominator.\n\nConverting 3/4 to twelfths: 3/4 = 9/12. So, 1/12 - 9/12 = (1 - 9)/12 = -8/12. Simplifying that fraction by dividing numerator and denominator by 4: -8/12 = -2/3.\n\nHmm, looking at the answer choices, option A is -2/3. So, is that the answer? Wait, but let me double-check my calculations to make sure I didn't make a mistake somewhere.\n\nFirst, checking (1⊗2): 1² / 2 = 1/2. Correct. Then, (1/2)⊗3: (1/2)² / 3 = (1/4)/3 = 1/12. That seems right.\n\nNow, for 2⊗3: 2² / 3 = 4/3. Correct. Then, 1⊗(4/3): 1² / (4/3) = 1 / (4/3) = 3/4. Yes, that's correct.\n\nSubtracting 3/4 from 1/12: 1/12 - 3/4. Convert 3/4 to 9/12, so 1/12 - 9/12 = -8/12 = -2/3. Yes, that all checks out. So the answer should be -2/3, which is option A.\n\nWait, but let me think again. The operation is defined for all nonzero numbers, so we don't have any issues with division by zero here. 2⊗3 is 4/3, which is fine, and then 1⊗(4/3) is 3/4. Correct.\n\nAlternatively, maybe there's a different way to approach the problem? Let me try expanding both expressions using variables to see if there's a pattern.\n\nLet’s denote the first expression: (a⊗b)⊗c. Using the definition:\n\nFirst, compute a⊗b = a²/b.\n\nThen, take that result and ⊗ with c: (a²/b)⊗c = ( (a²/b)² ) / c = a⁴ / (b² c).\n\nNow, the second expression: a⊗(b⊗c). First compute b⊗c = b²/c.\n\nThen, a⊗(b²/c) = a² / (b²/c) = a² * (c / b²) = (a² c) / b².\n\nTherefore, the difference between the two expressions is:\n\n(a⁴ / (b² c)) - (a² c / b²) = (a⁴ - a² c²) / (b² c) = a² (a² - c²) / (b² c).\n\nHmm, factoring that, it's a² (a - c)(a + c) / (b² c).\n\nBut in our specific problem, a = 1, b = 2, c = 3. Plugging those values in:\n\n1² (1 - 3)(1 + 3) / (2² * 3) = 1 * (-2)(4) / (4 * 3) = (-8) / 12 = -2/3. Same result. So that confirms the answer is indeed -2/3.\n\nTherefore, I think my initial calculation was correct, and the answer is option A.\n\n**Final Answer**\n\\boxed{A}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\nTo determine the value of \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) where the operation \\(\\otimes\\) is defined by \\(a \\otimes b = \\frac{a^2}{b}\\), we proceed as follows:\n\nFirst, compute \\(1 \\otimes 2\\):\n\\[\n1 \\otimes 2 = \\frac{1^2}{2} = \\frac{1}{2}\n\\]\nNext, use this result to compute \\((1 \\otimes 2) \\otimes 3\\):\n\\[\n\\left(\\frac{1}{2}\\right) \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n\\]\n\nNow, compute \\(2 \\otimes 3\\):\n\\[\n2 \\otimes 3 = \\frac{2^2}{3} = \\frac{4}{3}\n\\]\nThen, use this result to compute \\(1 \\otimes (2 \\otimes 3)\\):\n\\[\n1 \\otimes \\left(\\frac{4}{3}\\right) = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n\\]\n\nFinally, find the difference between the two results:\n\\[\n\\frac{1}{12} - \\frac{3}{4} = \\frac{1}{12} - \\frac{9}{12} = \\frac{1 - 9}{12} = \\frac{-8}{12} = -\\frac{2}{3}\n\\]\n\nThus, the answer is \\(\\boxed{A}\\).\n\n<|end_of_solution|>" + "<|im_end|>\n<|im_start|>user\n" + p['conversations'][0]['value'] + "<|im_end|>\n<|im_start|>assistant\n"

        original_prompt = prompt
        orpleen  = len(original_prompt)
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            stop_token_ids = tok("<|begin_of_thought|> <|end_of_thought|>")["input_ids"]
        else:
            stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=args.temperature,
            top_p = args.topp
        )
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            prompt += "<|begin_of_thought|> "
        else:
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
                top_p = args.topp,
            )
            o = model.generate(
                prompt,
                sampling_params=sampling_params
            )
        ### Final answer ###
        prompt += o[0].outputs[0].text # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
            stop_token_ids = tok("<|end_of_solution|>")["input_ids"]
        else: 
            stop_token_ids = tok("<|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=args.max_final_tokens,
            min_tokens=min(args.min_tokens, args.max_final_tokens),
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=args.temperature,
            top_p = args.topp,
        )
        o = model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        print("With budget forcing:") # You will see that after the "Wait" in the reasoning trace it fixes its answer
        if args.dataset == "bespokelabs/Bespoke-Stratos-17k":
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
                     'rejected' : prompt[orpleen+17:] + o[0].outputs[0].text
                     }

        data.append(entry)
        
        with open(f"{args.dataset[:11]}.json", "w") as outfile:
            json.dump(data, outfile, indent=4)

Inspired by s1 script

command line arguments:

--model_name:  model name, including vendor

--min_tokens:  minimal amount of tokens that the model is able to think in total

--max_tokens: token threshold, or maximal amount of tokens that we allow it to think during each call of the generate function

--max_final_tokens: maximum amount of tokens the model is allowed to output after its thinking was interrupted and it is forced to give a final answer. If equals to zero, we do not separate final answer from the rest
of the thinking process.

--temperature: generation temperature

--num_ignore: how many times to ignore end-of-thinking token

--dataset: dataset name. Currently supports only VanWang/NuminaMath-CoT_O1_Qwq and Open-COT-Data/COT-Dataset-Math

--ngpus: number of GPUs to parallelize upon. Defaults to 1.

--mode: Whether we want to keep generating until both num_ignore and min_tokens expire ("strict") or until one of them expires ("lax"), and then force it to generate answer

--custom_prompt: If we want to give some custom assistant prompt.

--cp_arg_1: Field, or name of the column in the dataset. Depends on the dataset. Introduced to allow arbitrary datasets. Can also be prompt

--cp_arg_2: Field, or name of the column in the dataset. Depends on the dataset. Introduced to allow arbitrary datasets. Can also be prompt

--cp_arg_3: Field, or name of the column in the dataset. Depends on the dataset. Introduced to allow arbitrary datasets. Can also be prompt

--custom_final_answer_prompt: if we want to steer the model by some custom answer towards a certain thoughts when generating the final answer.

--ignore_string: if we are not content on "wait" or "wait, think again", we can provide our own prompt that will force the model to backtrack.
Example:

```
python model_budgeting.py --model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--min_tokens=32000 --max_tokens=32000 --max_final_tokens=0 --temperature=0.0 \
--num_ignore=0 --dataset="bespokelabs/Bespoke-Stratos-17k" --ngpus=1 \
--topp=0.000000001 --mode="strict"
```


Currently supports only Numina-Math dataset

To get prompt steering, run file model_stats.py with the same arguments as model_budgeting.py (except dataset argument which you skip)


utils script and Frechet BERT distance computation procedure adopted from here

https://github.com/yhlleo/frechet-bert-distance

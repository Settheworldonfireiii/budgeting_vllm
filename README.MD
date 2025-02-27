Inspired by s1 script

command line arguments:

--model_name:  model name, including vendor

--min_tokens:  minimal amount of tokens that the model must think for during each generation, whether original one or extended

--max_tokens: token threshold, or maximal amount of tokens that we allow before we interrupt the model and force it to give the answer

--max_final_tokens: maximum amount of tokens the model is allowed to output after its thinking was interrupted and it is forced to give a final answer

--temperature: generation temperature

--num_ignore: how many times to ignore end-of-thinking token

--dataset: dataset name. Currently supports only VanWang/NuminaMath-CoT_O1_Qwq and Open-COT-Data/COT-Dataset-Math

Example:

```
python model_budgeting.py --model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
--min_tokens=400 --max_tokens=64000 --max_final_tokens=200 --temperature=0.0 \ 
--num_ignore=3 --dataset="Open-COT-Data/COT-Dataset-Math"
```


Currently supports only Numina-Math dataset

To get prompt steering, run file model_stats.py with the same arguments as model_budgeting.py (except dataset argument which you skip)

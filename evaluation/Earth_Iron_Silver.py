import json
import os
from datasets import load_dataset
from utils import LLM, muti_thread
from prompts import get_prompt
from show_results import score_format, score_task, score_category

# Set up the experiment parameters
ds_name = 'Earth-Silver' # 'Earth-Iron', 'Earth-Silver'
ques_type = 'free_form' # 'multiple_choice', 'true_false', 'fill_in_the_blank', 'free_form'
model_name = 'gpt-4o'

# Load the dataset and model
ds = load_dataset(f"ai-earth/{ds_name}")
llm = LLM(model_name)

# Prepare for input
prompts = []
for ques in ds[ques_type]:
    prompt = get_prompt(ques['question'], ques_type)
    prompts.append({'query':prompt})

# Multithreaded evaluation
print(f"Evaluating {ds_name}[{ques_type}] questions using {model_name}...")
llm_answers = muti_thread(prompts, llm.__call__, max_workers=50)

# Save the results
results = []
for idx, ques in enumerate(ds[ques_type]):
    result = dict(ques)
    result.update({'llm_answer': llm_answers[idx]})
    results.append(result)

# Calculate Win Rate and Semantic Similarity for free_form questions
if ques_type == 'free_form':
    print("Calculating Win Rate and Semantic Similarity...")
    from utils import FreeFormMetrics
    metrics = FreeFormMetrics()

    metrics_inp = []
    for result in results:
        metrics_inp.append({'ques_ans_dict': result})
    results = muti_thread(metrics_inp, metrics.eval, max_workers=50)

save_path = f'./results/{ds_name}/{model_name}/{ques_type}.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

score_format(ds_name)
score_task(ds_name)
score_category(ds_name)
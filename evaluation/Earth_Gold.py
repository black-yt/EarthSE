import os
import json
from datasets import load_dataset
from utils import LLM, muti_thread, multi_trun_dialogue, DialogueMetrics
from show_results import score_SES


# Set up the experiment parameters
model_name = 'gpt-4o'
m = 3

ds = load_dataset(f"ai-earth/Earth-Gold")
llm = LLM(model_name)

# Prepare for input
prompts = []
for ques in ds['train']:
    prompts.append({'dialogue_dict': dict(ques), 'llm': llm, 'm': m})

# Multithreaded evaluation
print(f"Evaluating Earth-Gold questions using {model_name}...")
results = muti_thread(prompts, multi_trun_dialogue, max_workers=50)

# Calculate Scientific Exploration Score
print("Calculating Scientific Exploration Score...")
metrics = DialogueMetrics()
metrics_inp = []
for result in results:
    metrics_inp.append({'dialogue_dict': result})
results = muti_thread(metrics_inp, metrics.eval, max_workers=50)

# Save the results
save_path = f'./results/Earth-Gold/{model_name}.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

score_SES()
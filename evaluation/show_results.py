import os
import json

def division(a, b):
    if b == 0:
        return 0
    else:
        return a / b
    
def mean(x):
    if len(x) == 0:
        return 0
    return sum(x) / len(x)

def score_format(ds_name, detail=False):
    ans_dir = f'./results/{ds_name}'

    print()
    if detail:
        print('|Model                          | multiple_choice  |true_false        |fill_in_the_blank |free_form (Acc.)  |free_form (SS)    |')
        print('|-------------------------------|------------------|------------------|------------------|------------------|------------------|')
    else:
        print('|Model                          |multiple|true_fal|fill_in |free Acc|free SS |')
        print('|-------------------------------|--------|--------|--------|--------|--------|')
    for model_version in sorted(os.listdir(ans_dir)):
        row = '|'+f"{model_version:<30} " + '|'
        for ques_type in ['multiple_choice', 'true_false', 'fill_in_the_blank', 'free_form']:
            if ques_type == 'free_form':
                ss_list = []
            if not os.path.exists(os.path.join(ans_dir, model_version, f'{ques_type}.json')):
                if ques_type == 'free_form':
                    if detail:
                        row += '                  |                  |'
                    else:
                        row += '        |        |'
                else:
                    if detail:
                        row += '                  |'
                    else:
                        row += '        |'
                continue

            with open(os.path.join(ans_dir, model_version, f'{ques_type}.json'), 'r', encoding='utf-8') as file:
                results = json.load(file)

            correct_ans = 0
            all_ans = 0

            for q in results:
                if 'llm_answer' in q:
                    all_ans += 1
                    if q['answer'] == q['llm_answer'] or ('win' in q and q['win'] == 'llm_answer'):
                        correct_ans += 1
                    if 'similarity' in q:
                        ss_list.append(q['similarity'])

            if ques_type == 'free_form':
                if detail:
                    row += f'{correct_ans:<4}/{all_ans:<4}={division(correct_ans, all_ans)*100:05.2f} % |{division(sum(ss_list), len(ss_list)):<5.2f}             |'
                else:
                    row += f'{division(correct_ans, all_ans)*100:05.2f}   |{division(sum(ss_list), len(ss_list)):<5.2f}   |'
            else:
                if detail:
                    row += f'{correct_ans:<4}/{all_ans:<4}={division(correct_ans, all_ans)*100:05.2f} % |'
                else:
                    row += f'{division(correct_ans, all_ans)*100:05.2f}   |'
        print(row)
    print()


def score_task(ds_name):
    ans_dir = f'./results/{ds_name}'

    print()
    print('|Model                          |knowledg|fact_che|analysis|calculat|term_exp|relation|tool_usa|literatu|dataset |experime|code_gen|')
    print('|-------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|')
    for model_version in sorted(os.listdir(ans_dir)):
        row = '|'+f"{model_version:<30} " + '|'
        task_right_all = {
            'knowledge_qa': {'right': 0, 'all': 0}, 
            'fact_checking': {'right': 0, 'all': 0}, 
            'analysis': {'right': 0, 'all': 0}, 
            'calculation': {'right': 0, 'all': 0}, 
            'term_explanation': {'right': 0, 'all': 0}, 
            'relationship_extraction': {'right': 0, 'all': 0}, 
            'tool_usage': {'right': 0, 'all': 0}, 
            'literature_listing': {'right': 0, 'all': 0}, 
            'dataset': {'right': 0, 'all': 0}, 
            'experiment_design': {'right': 0, 'all': 0}, 
            'code_generation': {'right': 0, 'all': 0}, 
        }

        for ques_type in ['multiple_choice', 'true_false', 'fill_in_the_blank', 'free_form']:
            if not os.path.exists(os.path.join(ans_dir, model_version, f'{ques_type}.json')):
                continue
            with open(os.path.join(ans_dir, model_version, f'{ques_type}.json'), 'r', encoding='utf-8') as file:
                results = json.load(file)

            for q in results:
                if 'llm_answer' in q:
                    task = q['task']
                    task_right_all[task]['all'] += 1
                    if q['answer'] == q['llm_answer'] or ('win' in q and q['win'] == 'llm_answer'):
                        task_right_all[task]['right'] += 1
        
        for task, right_all in task_right_all.items():
            row += f"{division(right_all['right'], right_all['all'])*100:05.2f}   |"
        print(row)
    print()


def score_category(ds_name, detail=False):
    ans_dir = f'./results/{ds_name}'

    print()
    if detail:
        print('|Model                          | Hydrosphere      | Biosphere        | Lithosphere      | Atmosphere       | Cryosphere       |')
        print('|-------------------------------|------------------|------------------|------------------|------------------|------------------|')
    else:
        print('|Model                          |Hydrosph|Biospher|Lithosph|Atmosphe|Cryosphe|')
        print('|-------------------------------|--------|--------|--------|--------|--------|')
    for model_version in sorted(os.listdir(ans_dir)):
        row = '|'+f"{model_version:<30} " + '|'
        category_right_all = {
            'Hydrosphere': {'right': 0, 'all': 0}, 
            'Biosphere': {'right': 0, 'all': 0}, 
            'Lithosphere': {'right': 0, 'all': 0}, 
            'Atmosphere': {'right': 0, 'all': 0}, 
            'Cryosphere': {'right': 0, 'all': 0}, 
        }

        for ques_type in ['multiple_choice', 'true_false', 'fill_in_the_blank', 'free_form']:
            if not os.path.exists(os.path.join(ans_dir, model_version, f'{ques_type}.json')):
                continue
            with open(os.path.join(ans_dir, model_version, f'{ques_type}.json'), 'r', encoding='utf-8') as file:
                results = json.load(file)

            for q in results:
                if 'llm_answer' in q:
                    category = q['sphere']
                    category_right_all[category]['all'] += 1
                    if q['answer'] == q['llm_answer'] or ('win' in q and q['win'] == 'llm_answer'):
                        category_right_all[category]['right'] += 1
        
        for category, right_all in category_right_all.items():
            if detail:
                row += f"{right_all['right']:<4}/{right_all['all']:<4}={division(right_all['right'], right_all['all'])*100:05.2f} % |"
            else:
                row += f"{division(right_all['right'], right_all['all'])*100:05.2f}   |"
        print(row)
    print()


def score_SES():
    print('|Model                          | retention_rate | diversity | SES    |')
    print('|-------------------------------|----------------|-----------|--------|')
    for model_version in os.listdir('./results/Earth-Gold'):
        with open(os.path.join('./results/Earth-Gold', model_version), 'r', encoding='utf-8') as file:
            results = json.load(file)
        diversity_list = []
        retention_rate_list = []
        SEC_list = []
        for ques in results:
            if 'diversity' in ques:
                diversity_list.append(ques['diversity'])
            if 'retention_rate' in ques:
                retention_rate_list.append(ques['retention_rate'])
            if 'SES' in ques:
                SEC_list.append(ques['SES'])
        print('|'+f"{model_version.replace('.json', ''):<30} | {mean(retention_rate_list)*100:05.2f} %        | {mean(diversity_list):06.4f}    | {mean(SEC_list):06.4f} |")


if __name__ == '__main__':
    ds_name = 'Earth-Iron' # 'Earth-Iron', 'Earth-Silver'
    score_format(ds_name, detail=False)
    score_task(ds_name)
    score_category(ds_name, detail=False)
    score_SES()

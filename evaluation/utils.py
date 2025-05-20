import os
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import torch
from sentence_transformers import SentenceTransformer, util

def muti_thread(inp_list, function, max_workers=40):
    results = [None] * len(inp_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(function, **item): index
            for index, item in enumerate(inp_list)
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing item {inp_list[index]}: {str(e)}")
    
    return results

class LLM:
    def __init__(self, model='gpt-4o', **kwargs):
        # export OPENAI_API_KEY="xxxxx"
        # export OPENAI_BASE_URL="xxxxx"
        self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY'))
        self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_BASE_URL'))
        self.model = model
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def __call__(self, query=None, **kwargs):
        system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
        max_tokens = kwargs.get('max_tokens', 4096)
        temperature = kwargs.get('temperature', 0)
        history = kwargs.get('history', None)

        if history is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        else:
            messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assistant_response = response.choices[0].message.content
        return assistant_response


class FreeFormMetrics:
    def __init__(self):
        self.judge_llm = LLM('gpt-4')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'similarity model device: {self.device}')
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

    def eval_win_rate(self, ques_ans_dict):
        prompt_temp = """
Please act as an expert evaluator and determine which of the following two answers is better.

**Evaluation Criteria:**
1. Assess how well each answer addresses the original question. Closer alignment is better.
2. Evaluate the scientific accuracy and logical coherence of each answer. More rigorous and professional reasoning is preferred.
3. Consider the relevance and depth of detail. More relevant and well-supported details indicate a better answer.
4. It is not the case that the longer the answer, the better. If the answer is long but does not meet the above requirements, it is not a good answer.

**Instructions:**
1. Do **not** generate a new answer to the original question. Your task is only to evaluate the two provided answers.
2. Based on the criteria above, choose which answer is better.
3. Your response must be **only** one letter: `A` or `B`.
4. Do **not** provide explanations, commentary, or corrections, even if there are errors in the inputs.
5. This is purely an evaluation task.

**[Question Start]**
<QUES>
**[Question End]**

**[Answer A Start]**
<ANS_A>
**[Answer A Start]**

**[Answer B Start]**
<ANS_B>
**[Answer B Start]**

**The better answer is:**
"""

        if 'win' in ques_ans_dict and ques_ans_dict['win'] in ['reference_answer', 'llm_answer']:
            return ques_ans_dict
        
        ques = ques_ans_dict['question']
        ans_A = ques_ans_dict['answer']
        ans_B = ques_ans_dict['llm_answer']
        prompt = prompt_temp.replace('<QUES>', str(ques)).replace('<ANS_A>', str(ans_A)).replace('<ANS_B>', str(ans_B))

        result = self.judge_llm(prompt)

        if result is not None:
            if result == 'A':
                ques_ans_dict['win'] = 'reference_answer'
            elif result == 'B':
                ques_ans_dict['win'] = 'llm_answer'
        return ques_ans_dict


    def eval_similarity(self, ques_ans_dict):
        if 'similarity' in ques_ans_dict and (isinstance(ques_ans_dict['similarity'], float) or isinstance(ques_ans_dict['similarity'], int)):
            return ques_ans_dict
        embedding_1 = self.similarity_model.encode(str(ques_ans_dict['answer']), convert_to_tensor=True)
        embedding_2 = self.similarity_model.encode(str(ques_ans_dict['llm_answer']), convert_to_tensor=True)
        ques_ans_dict['similarity'] = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return ques_ans_dict


    def eval(self, ques_ans_dict):
        ques_ans_dict = self.eval_win_rate(ques_ans_dict)
        ques_ans_dict = self.eval_similarity(ques_ans_dict)
        return ques_ans_dict



def multi_trun_dialogue(dialogue_dict, llm: LLM, m=3):
    prompt = 'Please respond to the following question in 80 words or less.\n'
    res_list = []

    for _ in range(m):
        res_1 = llm(prompt + dialogue_dict['user_0'], temperature=0.6)
        res_2 = llm(prompt + dialogue_dict['user_1'], history=[{"role": "user", "content": dialogue_dict['user_0']}, {"role": "assistant", "content": res_1}])
        res_list.append({
            'user_0': dialogue_dict['user_0'],
            'assistant_0': res_1,
            'user_1': dialogue_dict['user_1'],
            'assistant_1': res_2,
        })
    
    dialogue_dict['llm_answer'] = res_list
    return dialogue_dict


class DialogueMetrics:
    def __init__(self):
        self.judge_llm = LLM('gpt-4')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'similarity model device: {self.device}')
        self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

    def transformer_similarity(self, s_list):
        s_emb_list = []
        for s in s_list:
            s_emb = self.similarity_model.encode(s, convert_to_tensor=True)
            s_emb_list.append(s_emb)
        s_emb_mean = torch.mean(torch.stack(s_emb_list), dim=0)
        cos_list = []
        for s_emb in s_emb_list:
            cos = util.pytorch_cos_sim(s_emb_mean, s_emb)
            cos_list.append(cos.item())
        cos_mean = sum(cos_list) / len(cos_list)
        return cos_mean
    
    def eval_diversity(self, ques_ans_dict):
        if 'diversity' in ques_ans_dict and (isinstance(ques_ans_dict['diversity'], float) or isinstance(ques_ans_dict['diversity'], int)):
            return ques_ans_dict
        
        s_list = []
        for answers in ques_ans_dict['llm_answer']:
            s_list.append(answers['assistant_0']+answers['assistant_1'])
        similarity = self.transformer_similarity(s_list)
        similarity = (max(abs(similarity)-0.9, 0.01))*10
        ques_ans_dict['diversity'] = 1 / similarity
        return ques_ans_dict

    
    def eval_retention_rate(self, ques_ans_dict):
        prompt_temp = """
Please rank the reference dialogue among all dialogues based on the following criteria:

**Evaluation Criteria:**
1. Depth of Reflection: A dialogue is considered high-quality if it contains in-depth analysis and reflection on the topic.
2. Novelty of Approach: A dialogue is considered high-quality if it proposes innovative solutions or unique insights.

You only need to output **an integer** representing the ranking of the reference dialogue among all dialogues (1 being the best, higher numbers indicating lower rankings).

**[Reference Dialogue Start]**
<Dialogue_1>
**[Reference Dialogue End]**

**[Other Dialogues Start]**
<Dialogue_2>
**[Other Dialogues Start]**

**The ranking of the reference dialogue among all dialogues (An integer between 1 and <NUM>):**
"""

        if 'retention_rate' in ques_ans_dict and (isinstance(ques_ans_dict['retention_rate'], float) or isinstance(ques_ans_dict['retention_rate'], int)):
            return ques_ans_dict
        
        Dialogue_1 = {'user': ques_ans_dict['user_0'], 'assistant': ques_ans_dict['assistant_0'], 'user': ques_ans_dict['user_1'], 'assistant': ques_ans_dict['assistant_1']}
        Dialogue_2 = ques_ans_dict['llm_answer']
        Dialogue_2 = [str({'user': d['user_0'], 'assistant': d['assistant_0'], 'user': d['user_1'], 'assistant': d['assistant_1']}) for d in Dialogue_2]
        num = len(Dialogue_2)+1
        prompt = prompt_temp.replace('<Dialogue_1>', str(Dialogue_1)).replace('<Dialogue_2>', '\n\n'.join(Dialogue_2)).replace('<NUM>', str(num))

        result = None
        result = self.judge_llm(prompt).strip()
        try:
            assert result in [str(i) for i in range(1, num+1)]
            result = int(result)
        except:
            result = None

        if result is not None:
            ques_ans_dict['retention_rate'] = (result-1)/(num-1)
        return ques_ans_dict

    def eval(self, dialogue_dict):
        dialogue_dict = self.eval_retention_rate(dialogue_dict)
        dialogue_dict = self.eval_diversity(dialogue_dict)
        if 'diversity' in dialogue_dict and (isinstance(dialogue_dict['diversity'], float) or isinstance(dialogue_dict['diversity'], int)) and 'retention_rate' in dialogue_dict and (isinstance(dialogue_dict['retention_rate'], float) or isinstance(dialogue_dict['retention_rate'], int)):
            dialogue_dict['SES'] = dialogue_dict['diversity'] * dialogue_dict['retention_rate']
        return dialogue_dict


if __name__ == '__main__':
    llm = LLM()
    query = "hello!"
    response = llm(query)
    print(response)
def multiple_choice_prompt(ques: str):
    return f'''
Please respond to the following multiple-choice question by providing your answer as a single letter, without any additional text.

{ques}

The answer is (single letter):
'''

def true_false_prompt(ques: str):
    return f'''
Please answer the following true or false question with "True" or "False" without adding any additional text.

{ques}

The answer is ("True" or "False"):
'''

def fill_in_the_blank_prompt(ques: str):
    return f'''
Please answer the fill-in-the-blank question below with lowercase words or phrases. If your answer contains multiple words or phrases, please separate them with commas. No additional text is required.

{ques}

The answer is:
'''

def free_form_prompt(ques: str):
    return f'''
Please answer the following question:

{ques}

The answer is:
'''


def dialogue_prompt(ques: str):
    return f'''
'Please respond to the following question in 80 words or less.

{ques}
'''

def get_prompt(ques: str, type: str):
    if type == 'multiple_choice':
        return multiple_choice_prompt(ques)
    elif type == 'true_false':
        return true_false_prompt(ques)
    elif type == 'fill_in_the_blank':
        return fill_in_the_blank_prompt(ques)
    elif type == 'free_form':
        return free_form_prompt(ques)
    elif type == 'dialogue':
        return dialogue_prompt(ques)
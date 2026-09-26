import re
import numpy as np
SPECIAL = ['<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>']

def encode(tokenizer, state, question, length, max_options):

    def tokens(text):
        text = re.sub('<\\|([A-Za-z0-9_]+)\\|>', '<¦\\1¦>', text)
        return tokenizer.encode(text, add_special_tokens=False).ids
    s, q, o, end, decide = [tokenizer.token_to_id(t) for t in SPECIAL]
    if None in (s, q, o, end, decide):
        raise ValueError('tokenizer lacks Kev control tokens')
    options = question['options']
    if not 1 <= len(options) <= max_options:
        raise ValueError(f'question requires 1..{max_options} options')
    ids = [s] + tokens(state) + [q] + tokens(question['instr'])
    ends = []
    for option in options:
        ids += [o] + tokens(option) + [end]
        ends.append(len(ids) - 1)
    ids += [decide]
    if len(ids) > length:
        raise ValueError(f'state and question need {len(ids)} tokens; maximum is {length}; no text was truncated')
    inputs = {'input_ids': np.zeros((1, length), dtype=np.int64), 'decide_map': np.zeros((1, 1, length), dtype=np.float32), 'option_map': np.zeros((1, max_options, length), dtype=np.float32)}
    inputs['input_ids'][0, :len(ids)] = ids
    inputs['decide_map'][0, 0, len(ids) - 1] = 1
    inputs['option_map'][0, range(len(ends)), ends] = 1
    return (inputs, len(ids))

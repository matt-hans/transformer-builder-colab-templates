import sys, os, importlib.util, types
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Directly load the collator module by path to avoid heavy utils imports
dc_path = os.path.join(repo_root, 'utils', 'tokenization', 'data_collator.py')
spec = importlib.util.spec_from_file_location('data_collator', dc_path)
dc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dc)  # type: ignore
LanguageModelingDataCollator = dc.LanguageModelingDataCollator


class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.padding_side = 'right'

    def pad(self, examples, return_tensors=None, padding=True):
        max_len = max(len(ex['input_ids']) for ex in examples)
        out = {'input_ids': [], 'attention_mask': []}
        for ex in examples:
            ids = list(ex['input_ids'])
            pad_len = max_len - len(ids)
            if self.padding_side == 'left':
                padded = [self.pad_token_id] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            out['input_ids'].append(padded)
            out['attention_mask'].append(mask)
        return out


def test_dynamic_padding_right_side():
    tok = DummyTokenizer(pad_token_id=0)
    collator = LanguageModelingDataCollator(tok, mlm=False, padding_side='right')
    ex = [{'input_ids': [1,2,3]}, {'input_ids': [4,5]}]
    batch = collator(ex)
    assert batch['input_ids'] == [[1,2,3],[4,5,0]]
    assert batch['labels'] == [[1,2,3],[4,5,0]]
    assert batch['attention_mask'] == [[1,1,1],[1,1,0]]


def test_dynamic_padding_left_side():
    tok = DummyTokenizer(pad_token_id=0)
    collator = LanguageModelingDataCollator(tok, mlm=False, padding_side='left')
    ex = [{'input_ids': [7,8]},{'input_ids':[9]}]
    batch = collator(ex)
    assert batch['input_ids'] == [[7,8],[0,9]]
    assert batch['labels'] == [[7,8],[0,9]]
    assert batch['attention_mask'] == [[1,1],[0,1]]

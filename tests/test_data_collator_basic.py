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


def test_tensor_compatibility():
    """Test that collator handles PyTorch tensors (HuggingFace tokenizer behavior)."""
    import torch

    class TensorTokenizer:
        """Mock HuggingFace tokenizer that returns tensors even with return_tensors=None."""
        def __init__(self, pad_token_id=0):
            self.pad_token_id = pad_token_id
            self.padding_side = 'right'

        def pad(self, examples, return_tensors=None, padding=True):
            """Simulates HuggingFace behavior: preserves tensor type from input."""
            max_len = max(len(ex['input_ids']) for ex in examples)
            out_ids = []
            out_mask = []
            for ex in examples:
                # If input is tensor, convert to list for padding, then back to tensor
                if torch.is_tensor(ex['input_ids']):
                    ids = ex['input_ids'].tolist()
                else:
                    ids = list(ex['input_ids'])

                pad_len = max_len - len(ids)
                if self.padding_side == 'left':
                    padded = [self.pad_token_id] * pad_len + ids
                    mask = [0] * pad_len + [1] * len(ids)
                else:
                    padded = ids + [self.pad_token_id] * pad_len
                    mask = [1] * len(ids) + [0] * pad_len

                # Return as tensors (HuggingFace behavior)
                out_ids.append(torch.tensor(padded))
                out_mask.append(torch.tensor(mask))

            return {'input_ids': out_ids, 'attention_mask': out_mask}

    # Test with tensor examples (simulates pre-tokenized tensor data from Cell 19)
    tok = TensorTokenizer(pad_token_id=0)
    collator = LanguageModelingDataCollator(tok, mlm=False, padding_side='right')

    # Examples with tensors (like what HuggingFace tokenizers produce)
    ex = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])}
    ]

    # Should not raise AttributeError: 'Tensor' object has no attribute 'copy'
    batch = collator(ex)

    # Verify results
    assert len(batch['input_ids']) == 2
    assert len(batch['labels']) == 2
    assert torch.is_tensor(batch['input_ids'][0])
    assert torch.is_tensor(batch['labels'][0])
    assert batch['input_ids'][0].tolist() == [1, 2, 3]
    assert batch['input_ids'][1].tolist() == [4, 5, 0]
    assert batch['labels'][0].tolist() == [1, 2, 3]
    assert batch['labels'][1].tolist() == [4, 5, 0]

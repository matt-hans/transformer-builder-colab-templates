"""
Data collators for variable-length sequences.

Provides a lightweight LanguageModelingDataCollator that performs dynamic
padding, builds attention masks, and supports causal (GPT) and masked (BERT)
objectives without requiring transformers at import time.
"""

from typing import List, Dict, Any, Optional


class LanguageModelingDataCollator:
    """Custom data collator for transformer language modeling.

    - Causal LM (default): labels = input_ids (model performs shift internally)
    - Masked LM: masks tokens with given probability when tokenizer supports it
    - Dynamic padding with optional left/right side
    """

    def __init__(self,
                 tokenizer: Any,
                 mlm: bool = False,
                 mlm_probability: float = 0.15,
                 padding_side: str = 'right'):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.padding_side = padding_side

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Use tokenizer.pad when available
        batch = None
        if hasattr(self.tokenizer, 'pad'):
            # Temporarily set padding_side if supported
            original_side = getattr(self.tokenizer, 'padding_side', None)
            try:
                if original_side is not None:
                    self.tokenizer.padding_side = self.padding_side
                batch = self.tokenizer.pad(
                    examples,
                    return_tensors=None,  # leave as lists; downstream will cast to torch if available
                    padding=True,
                )
            finally:
                if original_side is not None:
                    self.tokenizer.padding_side = original_side
        else:
            batch = self._pad_examples(examples)

        # Ensure attention_mask exists
        if 'attention_mask' not in batch:
            batch['attention_mask'] = self._build_attention_mask(batch['input_ids'])

        if not self.mlm:
            # labels same as input_ids for causal LM
            # (model performs shifting internally)
            batch['labels'] = [seq.copy() for seq in batch['input_ids']]
        else:
            input_ids = batch['input_ids']
            labels, masked_inputs = self._mask_tokens(input_ids)
            batch['labels'] = labels
            batch['input_ids'] = masked_inputs

        return batch

    def _pad_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Determine pad token id
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        max_len = 0
        for ex in examples:
            max_len = max(max_len, len(ex['input_ids']))

        result_ids: List[List[int]] = []
        for ex in examples:
            ids = list(ex['input_ids'])
            pad_len = max_len - len(ids)
            if self.padding_side == 'left':
                padded = [pad_id] * pad_len + ids
            else:
                padded = ids + [pad_id] * pad_len
            result_ids.append(padded)

        return {'input_ids': result_ids}

    def _build_attention_mask(self, input_ids: List[List[int]]) -> List[List[int]]:
        masks: List[List[int]] = []
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        for seq in input_ids:
            masks.append([0 if tok == pad_id else 1 for tok in seq])
        return masks

    def _mask_tokens(self, input_ids: List[List[int]]) -> (List[List[int]], List[List[int]]):
        # Simple masking strategy if tokenizer supports mask_token_id
        import random
        mask_id = getattr(self.tokenizer, 'mask_token_id', None)
        if mask_id is None:
            # fallback: no masking; labels = input_ids
            return ([s.copy() for s in input_ids], [s.copy() for s in input_ids])

        labels: List[List[int]] = []
        masked_inputs: List[List[int]] = []
        special_ids = set(getattr(self.tokenizer, 'all_special_ids', []) or [])
        for seq in input_ids:
            lbl = seq.copy()
            inp = seq.copy()
            for i, tok in enumerate(seq):
                if tok in special_ids:
                    continue
                if random.random() < self.mlm_probability:
                    inp[i] = mask_id
                else:
                    lbl[i] = -100  # ignore index for loss
            labels.append(lbl)
            masked_inputs.append(inp)
        return labels, masked_inputs


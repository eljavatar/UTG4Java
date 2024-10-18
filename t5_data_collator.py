'''
Note that this code mostly comes from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py with some modifications.
The code here doesn't rely on flax anymore.
https://github.com/huggingface/olm-training/blob/main/t5_data_collator.py

https://ychai.uk/notes/2022/01/10/Mask-Denoising-Strategy-for-Pre-trained-Models/
'''

from typing import Dict, List
import numpy as np
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
    BertTokenizer,
    BertTokenizerFast
)
from dataclasses import dataclass
import torch
import random
import warnings


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py#L59
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def compute_t5_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1

        #print(f"_input_length = {_input_length}, _output_length = {_output_length}")
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    #print("Final results:")
    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    #tokens_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    #print(f"_input_length = {tokens_length}, targets_length = {targets_length}")

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length




@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        
        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        #len_batch_inputs = batch["input_ids"].shape[-1]
        #len_batch_labels = batch["labels"].shape[-1]
        #print(f"\nLength len_batch_inputs: {len_batch_inputs}")
        #print(batch["input_ids"])
        #print(f"\nLength len_batch_labels: {len_batch_labels}")
        #print(batch["labels"])

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        for key, value in batch.items():
            batch[key] = torch.tensor(value)

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]





@dataclass
class DataCollatorWithDynamicLengthForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    max_length: int
    pad_token_id: int
    decoder_start_token_id: int
    whole_word_mask: bool = False

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        prints = []
        prints.append(f"\nEXAMPLES SIZE:\n")
        prints.append(examples)

        """
        if self.whole_word_mask:
            mask_labels = []
            for e in examples:
                ref_tokens = []
                for id in tolist(e["input_ids"]):
                    token = self.tokenizer._convert_id_to_token(id)
                    ref_tokens.append(token)

                # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
                if "chinese_ref" in e:
                    ref_pos = tolist(e["chinese_ref"])
                    len_seq = len(e["input_ids"])
                    for i in range(len_seq):
                        if i in ref_pos:
                            ref_tokens[i] = "##" + ref_tokens[i]
                mask_labels.append(self._whole_word_mask(ref_tokens))
        """

        batch = self.split_examples(examples)
        prints.append(f"\nBATCH SIZE:\n")
        prints.append(batch)
        prints.append(f"\n\n\n")
        #print(f"\n\n{prints}\n\n")

        for key, value in batch.items():
            try:
                batch[key] = torch.tensor(value, dtype=torch.int64)
            except (RuntimeError, Exception) as e:
                #print(f"\n\n\n\n\n********** CONVIRTIENDO {key} A TENSOR VALUE: **********\n{value}\n\n\n\n\n")
                raise e

        return batch


    def split_examples(self, list_examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        #split_inputs = [{'input_ids': ids} for ids in list_examples['input_ids']]
        #split_inputs[0]['input_ids']

        #print("\n\n")
        #print(list_examples)
        #print("\n\n")

        inputs = []
        attentions = []
        decoder_inputs = []
        decoder_attentions = []
        labels = []

        for index in range(len(list_examples)):
            example = [list_examples[index]]
            # convert list to dict and tensorize input
            batch = BatchEncoding(
                {k: np.array([example[i][k] for i in range(len(example))]) for k, v in example[0].items()}
            )

            input_ids = batch["input_ids"]
            batch_size, input_length = input_ids.shape

            mask_indices = np.asarray([self.random_spans_noise_mask(input_length) for i in range(batch_size)])
            #print(f"\nLength mask_indices: {len(mask_indices[0])}")
            #print(mask_indices)

            #mask_indices = np.array([np.pad(mask, (0, expandend_input_length - len(mask)), 'constant', constant_values=False) for mask in mask_indices])
            #print(f"\nLength mask_indices after: {len(mask_indices[0])}")

            labels_mask = ~mask_indices
            #print(f"\nLength labels_mask: {len(labels_mask[0])}")
            #print(labels_mask)

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

            #print(f"\nLength input_ids_sentinel: {len(input_ids_sentinel[0])}")
            #print(input_ids_sentinel)
            #print(f"\nLength labels_sentinel: {len(labels_sentinel[0])}")
            #print(labels_sentinel)
            
            batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
            batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

            len_batch_inputs = batch["input_ids"].shape[-1]
            len_batch_labels = batch["labels"].shape[-1]
            #print(f"\nLength len_batch_inputs: {len_batch_inputs}")
            #print(batch["input_ids"])
            #print(f"\nLength len_batch_labels: {len_batch_labels}")
            #print(batch["labels"])

            if len_batch_inputs > self.max_length:
                raise ValueError(
                    f"`input_ids` are incorrectly preprocessed. `input_ids` length is {len_batch_inputs}, but"
                    f" should be max {self.max_length}."
                )
            
            #decoded_inputs = self.tokenizer.batch_decode(batch["input_ids"])
            #decoded_labels = self.tokenizer.batch_decode(batch["labels"])
            #print(f"\nDecoded inputs: \n{decoded_inputs}")
            #print(f"\nDecoded labels: \n{decoded_labels}")

            # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
            batch["decoder_input_ids"] = shift_tokens_right(
                batch["labels"], self.pad_token_id, self.decoder_start_token_id
            )

            # Complete inputs until max_length
            if len_batch_inputs < self.max_length:
                pad_len = self.max_length - len_batch_inputs
                batch["input_ids"] = np.array([np.pad(item, (0, pad_len), 'constant', constant_values=self.pad_token_id) for item in batch["input_ids"]])
                #print("\nNew inputs:")
                #print(batch["input_ids"])
            
            attention_mask = np.zeros_like(batch["input_ids"])
            attention_mask[:, 0:len_batch_inputs] = 1
            batch["attention_mask"] = attention_mask
            #print(f"\nLength attention_mask: {batch['attention_mask'].shape[-1]}")
            #print(batch["attention_mask"])
            batch["decoder_attention_mask"] = np.ones_like(batch["decoder_input_ids"])

            inputs.append(np.squeeze(batch["input_ids"]))
            attentions.append(np.squeeze(batch["attention_mask"]))
            decoder_inputs.append(np.squeeze(batch["decoder_input_ids"]))
            decoder_attentions.append(np.squeeze(batch["decoder_attention_mask"]))
            labels.append(np.squeeze(batch["labels"]))

        
        max_length_labels = max(len(arr) for arr in labels)
        #labels_with_padding = [np.pad(arr, (0, max_length_labels - len(arr)), 'constant', constant_values=self.pad_token_id) for arr in labels]
        #labels_with_padding = ([(l if l != self.pad_token_id else -100) for l in label] for label in labels_with_padding)
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels_with_padding = [np.pad(arr, (0, max_length_labels - len(arr)), 'constant', constant_values=-100) for arr in labels]
        #print(f"\n\nLIST LABELS {labels[0][1]}:\n{labels}\n\nNEW LIST LABELS:\n{labels_with_padding}\n\n")

        max_length_decoder = max(len(arr) for arr in decoder_inputs)
        decoder_with_padding = [np.pad(arr, (0, max_length_decoder - len(arr)), 'constant', constant_values=self.pad_token_id) for arr in decoder_inputs]
        decoder_attentions_with_padding = [np.pad(arr, (0, max_length_decoder - len(arr)), 'constant', constant_values=self.pad_token_id) for arr in decoder_attentions]
        #print(f"\n\nLIST DECODER {decoder_inputs[0][2]}:\n{decoder_inputs}\n\nNEW LIST DECODER:\n{decoder_with_padding}\n\n")

        return {
            "input_ids": np.stack(inputs),
            "attention_mask": np.stack(attentions),
            "decoder_input_ids": np.stack(decoder_with_padding),
            "decoder_attention_mask": np.stack(decoder_attentions_with_padding),
            "labels": np.stack(labels_with_padding)
        }


    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_nonnoise_tokens = length - num_noise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))
        #num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        #num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length
        
        #print(f"\n\nlength: {length}")
        #print(f"orig_length: {orig_length}")
        #print(f"num_noise_spans: {num_noise_spans}") # Cantidad de tramos (spans)
        #print(f"num_noise_tokens: {num_noise_tokens}") # Cantidad de tokens a enmascarar
        #print(f"num_nonnoise_tokens: {num_nonnoise_tokens}") # Cantidad de tokens sin enmascarar

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        #print(f"noise_span_lengths: {noise_span_lengths}") # Tamaño de cada tramo a enmascarar
        #print(f"nonnoise_span_lengths: {nonnoise_span_lengths}") # Tamaño de los tramos sin enmascarar

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )

        #print("interleaved_span_lengths:")
        #print(interleaved_span_lengths)
        
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
    

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            #if token.startswith("##"):
            #    print(f"Token starts with ##: {token}")

            # Obtenemos los índices de los tokens que no son palabras completas
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        """
        if len(input_tokens) != len(cand_indexes):
            print("\n******************************")
            print(f"\n\n\n\n\ninput_tokens lenght: {len(input_tokens)}")
            print(input_tokens)
            print(f"\ncand_indexes length: {len(cand_indexes)}")
            print(cand_indexes)
            print("\n\n\n\n\n")
            print("\n******************************")
        return cand_indexes
        """
        
        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels
        
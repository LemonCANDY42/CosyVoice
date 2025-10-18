# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

import queue
import random
import logging
from typing import List

import numpy as np
import torch

IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


class TrtContextWrapper:
    """TensorRT Context Pool Wrapper with timeout and health monitoring.
    
    Manages a pool of TensorRT execution contexts to prevent resource leakage
    and provides timeout mechanism to avoid permanent blocking.
    """
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0', acquire_timeout=30):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.max_concurrent = trt_concurrent
        self.acquire_timeout = acquire_timeout
        self.device = device
        self.acquired_count = 0
        
        logging.info(f"Initializing TrtContextWrapper with {trt_concurrent} contexts on {device}")
        
        for idx in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert trt_context is not None, f'failed to create trt context #{idx}, maybe not enough CUDA memory, try reduce current trt concurrent {trt_concurrent}'
            self.trt_context_pool.put([trt_context, trt_stream])
        
        assert self.trt_context_pool.empty() is False, 'no available estimator context'
        logging.info(f"TrtContextWrapper initialized successfully with {trt_concurrent} contexts")

    def acquire_estimator(self):
        """Acquire a TensorRT context from the pool with timeout.
        
        Returns:
            tuple: ([context, stream], engine)
            
        Raises:
            RuntimeError: If timeout occurs or pool is exhausted
        """
        try:
            available_before = self.trt_context_pool.qsize()
            if available_before == 0:
                logging.error(f"TRT context pool is empty! Waiting for available context... (timeout={self.acquire_timeout}s)")
            
            result = self.trt_context_pool.get(timeout=self.acquire_timeout)
            self.acquired_count += 1
            
            # available_after = self.trt_context_pool.qsize()
            # if available_after <= 1:
            #     logging.warning(f"TRT context pool running low: {available_after}/{self.max_concurrent} available")
            
            return result, self.trt_engine
            
        except queue.Empty:
            error_msg = (f"Failed to acquire TRT context after {self.acquire_timeout}s timeout. "
                        f"Available: {self.trt_context_pool.qsize()}/{self.max_concurrent}. "
                        f"This likely indicates a resource leak - check that all acquired contexts are properly released.")
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def release_estimator(self, context, stream):
        """Release a TensorRT context back to the pool.
        
        Args:
            context: TensorRT execution context
            stream: CUDA stream
        """
        try:
            self.trt_context_pool.put([context, stream], block=False)
            self.acquired_count -= 1
        except queue.Full:
            logging.error("Failed to release TRT context - pool is full! This should never happen.")
            # 这种情况理论上不应该发生，但如果发生了，至少记录下来
            
    def health_check(self):
        """Check the health status of the context pool.
        
        Returns:
            dict: Health status information
        """
        available = self.trt_context_pool.qsize()
        status = {
            'available': available,
            'total': self.max_concurrent,
            'in_use': self.max_concurrent - available,
            'healthy': available > 0
        }
        
        if available == 0:
            logging.warning(f"TRT context pool health check FAILED: {status}")
        
        return status

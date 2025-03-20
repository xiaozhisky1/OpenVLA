"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers  生成离散化区间
        self.bins = np.linspace(min_action, max_action, self.n_bins) # 均匀划分 min_action ~ max_action 为 n_bins 个区间
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0   # 计算区间中心点，用于反向解码（decode_token_ids_to_actions()）# 注意：区间有 256 个边界点，但只有 255 个中心点。

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        # 设定动作 Token 开始索引，最后 256 个 Token 专门用于表示动作值
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        """将连续动作转换为 Token ID，并映射到 Token"""

        # 把 action 限制在 [min_action, max_action] 之间
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins) # 找到 action 属于哪一个 bins 区间（返回 1 ~ 256 之间的索引）

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action)) # 映射到最后 n_bins=256 个 Token。
            # self.tokenizer.decode()：将 token_id 转换为 Token 字符串
        else: # 处理批量数据
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs. 把 Token ID 还原成连续动作（反向操作）

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids # 还原 discretized_actions（Token ID → Bin Index）
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        # discretized_actions - 1：让索引从 1~256 调整为 0~255（Python 从 0 开始索引）；np.clip(..., 0, 255-1)：确保不会越界

        return self.bin_centers[discretized_actions] # bin_centers：存储了所有离散化区间的中心值。将索引映射到真实的连续动作值

    @property
    def vocab_size(self) -> int:
        return self.n_bins

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.utils import get_global_args, max_alloc_seq_len
from .base import DSAIndexer

from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    blockfp8_index_score_ragged_q_paged_k_dsv32,
    blockfp8_index_score_ragged_q_dense_k_dsv32,
)
from .bf16_backend import BF16Indexer


def _validate_torch_bf16_indexer_config(args):
    if args.infer.cache_type != "paged":
        raise ValueError(
            f"indexer_type=torch_bf16 only supports cache_type=paged, but got {args.infer.cache_type}"
        )


class TorchIndexer(DSAIndexer):
    impl = "torch"

    def append_indexer_kv(
        self,
        k_fp8,
        k_scale,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
        k_append: Optional[torch.Tensor] = None,
    ):
        """Append this step's indexer K (and scale) to the KV cache once."""
        delta_pos = seq_len_delta.delta_position_ids_tensor_device
        delta_seq = seq_len_delta.delta_seq_ids_tensor_device
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_k"],
                cache_accessor.block_table,
                k_fp8,
                delta_pos,
                delta_seq,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_ks"],
                cache_accessor.block_table,
                k_scale,
                delta_pos,
                delta_seq,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_k"], k_fp8, delta_pos, delta_seq
            )
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_ks"], k_scale, delta_pos, delta_seq
            )
        else:
            raise NotImplementedError()

    def blockfp8_index_score_dsa_torch_or_triton(
        self,
        q_fp8,
        weights,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal: bool = True,
    ):
        softfp8 = (
            getattr(get_global_args().infer, "raise_lower_bit_float_to", None)
            == "bfloat16"
        )

        if isinstance(cache_accessor, PagedKVCacheAccessor):
            return blockfp8_index_score_ragged_q_paged_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                k_page_table=cache_accessor.block_table,
                static_max_n=max_alloc_seq_len(get_global_args().infer.max_seq_len),
                causal=is_causal,
                softfp8=softfp8,
                impl=self.impl,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            return blockfp8_index_score_ragged_q_dense_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                causal=is_causal,
                softfp8=softfp8,
                impl=self.impl,
            )
        else:
            raise NotImplementedError()

    def _index_score(
        self,
        q,
        weights,
        seq_len_delta,
        cache_accessor,
        is_causal=True,
        ke=None,
        ks=None,
    ):
        return self.blockfp8_index_score_dsa_torch_or_triton(
            q, weights, seq_len_delta, cache_accessor, is_causal
        )


class TorchBF16Indexer(BF16Indexer):
    impl = "torch_bf16"

    def _decode_score(
        self,
        q,
        weights,
        seq_len_delta,
        cache_accessor,
        *,
        page_table=None,
        context_lens=None,
        score_width=None,
    ):
        scores = super()._decode_score(
            q,
            weights,
            seq_len_delta,
            cache_accessor,
            page_table=page_table,
            context_lens=context_lens,
            score_width=score_width,
        )
        # Ordinary Torch Decode returns BF16; local candidates keep FP32 scores.
        return scores.to(q.dtype) if context_lens is None else scores

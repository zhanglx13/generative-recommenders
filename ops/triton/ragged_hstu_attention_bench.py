import contextlib
import os

import click

import torch
import triton
import pytest
# from generative_recommenders.ops.triton.triton_ragged_hstu_attention import (
#     _RaggedAttentionFunction,
#     _RaggedAttentionRelativeBiasFunction,
# )

from triton_ragged_hstu_attention import (
    _RaggedAttentionFunction,
    _RaggedAttentionRelativeBiasFunction,
)

from triton_ragged_hstu_attention_ref import _RaggedAttentionFunction as _RaggedAttentionFunction_ref
from triton_ragged_hstu_attention_ref import _RaggedAttentionRelativeBiasFunction as _RaggedAttentionRelativeBiasFunction_ref

from torch.profiler import profile, ProfilerActivity


def profiler_or_nullcontext(enabled: bool, with_stack: bool):  # pyre-ignore [3]
    def _kineto_trace_handler(p: torch.profiler.profile) -> None:
        trace_url = "/tmp/libkineto_activities_{}.json".format(
            os.getpid(),
        )
        print(
            p.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total"
            )
        )
        print(f"trace url: {trace_url}")

        p.export_chrome_trace(trace_url)

    return (
        profile(
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=_kineto_trace_handler,
            with_stack=with_stack,
            record_shapes=True,
        )
        if enabled
        else contextlib.nullcontext()
    )


def generate_hstu_timestamps(batch_size: int, seq_len: int) -> torch.Tensor:
    # -0.8, 86400 //5 and 4*10e6 are hand-tuned parameters based on the distribution of standalone model
    timestamp_day: torch.Tensor = (
        torch.pow(
            torch.rand(size=(batch_size, seq_len + 1), device="cuda"), exponent=-0.8
        ).int()
        - 1
    ).int()
    timestamp_second: torch.Tensor = torch.normal(
        mean=torch.zeros(
            size=(batch_size, seq_len + 1),
            device="cuda",
        ),
        std=86400 // 5,
    ).int()
    timestamps = torch.clamp(
        torch.abs(timestamp_day * 86400 + timestamp_second), max=4 * 10e6
    )
    timestamps, _ = torch.sort(timestamps, dim=1)
    return timestamps.long()


def gen_inputs(dtype, attn_dim, batch_size, actual_seq_len, target_size, heads, hidden_dim, num_buckets, max_pos_ind):
    lengths = torch.zeros(batch_size, device=torch.device("cuda")) + actual_seq_len
    lengths = lengths + target_size
    num_targets = torch.randint(
        target_size, size=(batch_size,), device=torch.device("cuda")
    )
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads * (2 * attn_dim + 2 * hidden_dim)),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    u, v, q, k = torch.split(
        x,
        [hidden_dim * heads, hidden_dim * heads, attn_dim * heads, attn_dim * heads],
        dim=-1,
    )
    q = q.view(-1, heads, attn_dim)
    k = q.view(-1, heads, attn_dim)
    v = q.view(-1, heads, hidden_dim)
    # print(f"x = {x.shape}, q = {q.shape}, k = {k.shape}, v = {k.shape}")
    # print(f"q_stride = ({q.stride(0)}, {q.stride(1)}, {q.stride(2)})")
    # print(f"k_stride = ({k.stride(0)}, {k.stride(1)}, {v.stride(2)})")
    # print(f"v_stride = ({v.stride(0)}, {v.stride(1)}, {v.stride(2)})")

    ts_weights: torch.Tensor = (
        torch.empty(
            (num_buckets + 1,),
            device="cuda",
            dtype=torch.float32,
        )
        .uniform_(-0.1, 0.1)
        .to(dtype=torch.bfloat16)
    )
    pos_weights: torch.Tensor = (
        torch.empty(
            (2 * max_pos_ind - 1,),
            device="cuda",
            dtype=torch.float32,
        )
        .uniform_(-0.1, 0.1)
        .to(dtype=torch.bfloat16)
    )

    return lengths, num_targets, seq_offsets, u, v, q, k, ts_weights, pos_weights


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=32,
)
@click.option("--heads", type=int, default=4)
@click.option("--attn-dim", type=int, default=128)
@click.option("--hidden-dim", type=int, default=128)
@click.option("--max-seq-len", type=int, default=2550)
@click.option("--actual-seq-len", type=int, default=1616)
@click.option("--target-size", type=int, default=512)
@click.option("--max-pos-ind", type=int, default=4086)
@click.option("--no-relative-bias", is_flag=True, show_default=True, default=False)
@click.option("--profiling-mode", is_flag=True, show_default=True, default=False)
@click.option("--profiling-with-stack", is_flag=True, show_default=True, default=False)
def main(
    batch_size: int,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    max_seq_len: int,  # for repro
    actual_seq_len: int,  # for repro
    target_size: int,
    max_pos_ind: int,
    no_relative_bias: bool,
    profiling_mode: bool,
    profiling_with_stack: bool,
):
    dtype = torch.bfloat16
    warmup = 25
    rep = 1000
    alpha = 1.0 / (attn_dim**0.5)
    invalid_attn_mask_type = "lower_triangular"

    causal = True
    time_delta = 0.01
    num_buckets = 2048
    time_bucket_fn = "sqrt"
    time_bucket_incr = 60
    time_bucket_div = 1.0

    lengths, num_targets, seq_offsets, u, v, q, k, ts_weights, pos_weights = gen_inputs(dtype, attn_dim, batch_size, actual_seq_len, target_size, heads, hidden_dim, num_buckets, max_pos_ind)
    timestamps = generate_hstu_timestamps(batch_size, int(lengths.max().item()))
    relative_bias_type = "ALL"

    Z = seq_offsets.numel() - 1
    for i in range(Z):
        seq_offsets[i] = i*2048

    if not no_relative_bias:
        fn = lambda: _RaggedAttentionRelativeBiasFunction.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            timestamps,
            ts_weights,
            pos_weights,
            causal,
            num_buckets,
            time_bucket_fn,
            time_bucket_incr,
            time_bucket_div,
            time_delta,
            max_pos_ind,
            num_targets,
            None,
            relative_bias_type,
        )
    else:
        fn = lambda: _RaggedAttentionFunction.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            num_targets,
            None,
            None,
            None,
        )

    quantiles = [0.5, 0.2, 0.8]
    with profiler_or_nullcontext(profiling_mode, profiling_with_stack):
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn, warmup=warmup, rep=rep, quantiles=quantiles
        )
    print(
        f"P50 latency is {ms:.5f} ms\n"
        f"P20 latency is {min_ms:.5f} ms\n"
        f"P80 latency is {max_ms:.5f} ms"
    )


if __name__ == "__main__":
    main()


@pytest.mark.parametrize("batch_size, heads, attn_dim, hidden_dim, max_seq_len, actual_seq_len, target_size, max_pos_ind, no_relative_bias",
                         [(32, 4, 128, 128, 2550, 1616, 512, 4086, True),
                         (32, 4, 128, 128, 1264, 745, 512, 1024, True),
                         (32, 4, 128, 128, 2550, 1616, 512, 4086, False),
                         (32, 4, 128, 128, 1264, 745, 512, 1024, False)],)
def test_correctness(
    batch_size: int,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    max_seq_len: int,  # for repro
    actual_seq_len: int,  # for repro
    target_size: int,
    max_pos_ind: int,
    no_relative_bias: bool,
):

    dtype = torch.bfloat16
    alpha = 1.0 / (attn_dim**0.5)
    invalid_attn_mask_type = "lower_triangular"
    num_buckets = 2048

    lengths, num_targets, seq_offsets, u, v, q, k, ts_weights, pos_weights = gen_inputs(dtype, attn_dim, batch_size, actual_seq_len, target_size, heads, hidden_dim, num_buckets, max_pos_ind)

    causal = True
    time_delta = 0.01
    time_bucket_fn = "sqrt"
    time_bucket_incr = 60
    time_bucket_div = 1.0
    timestamps = generate_hstu_timestamps(batch_size, int(lengths.max().item()))
    relative_bias_type = "ALL"

    if not no_relative_bias:
        out_ref = _RaggedAttentionRelativeBiasFunction_ref.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            timestamps,
            ts_weights,
            pos_weights,
            causal,
            num_buckets,
            time_bucket_fn,
            time_bucket_incr,
            time_bucket_div,
            time_delta,
            max_pos_ind,
            num_targets,
            None,
            relative_bias_type,
        )

        out = _RaggedAttentionRelativeBiasFunction.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            timestamps,
            ts_weights,
            pos_weights,
            causal,
            num_buckets,
            time_bucket_fn,
            time_bucket_incr,
            time_bucket_div,
            time_delta,
            max_pos_ind,
            num_targets,
            None,
            relative_bias_type,
        )

    else:
        out_ref = _RaggedAttentionFunction_ref.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            num_targets,
            None,
            None,
            None,
        )

        out = _RaggedAttentionFunction.apply(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            num_targets,
            None,
            None,
            None,
        )

    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=0)

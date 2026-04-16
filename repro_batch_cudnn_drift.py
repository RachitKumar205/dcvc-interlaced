#!/usr/bin/env python3
"""
Minimal repro ladder for the batched fp16/cuDNN drift bug.

This script is intended to live on top of commit 5810653, which contains the
original 360p batching experiment setup but predates later debugging scripts
and fixes.

Levels:
  - full: run SEQ, B1, and B2 over a short sequence
  - encode: seed one I-frame, then compare one P-frame only
  - feature-extractor: compare the same reference sample at B=1 vs B=2

The `--guard on` pathway emulates the later fix locally by running the selected
subgraph under cuDNN-disabled fp16 fallback, without modifying model code.
"""

import argparse
from contextlib import contextmanager, nullcontext
import io

import torch
import torch.nn.functional as F

from src.models.image_model import DMCI
from src.models.video_model import DMC, RefFrame
from src.utils.common import get_state_dict, set_torch_env
from src.utils.transforms import ycbcr420_to_444_np
from src.utils.video_reader import YUV420Reader


@contextmanager
def cudnn_runtime(enabled):
    prev_enabled = torch.backends.cudnn.enabled
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.enabled = enabled
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.enabled = prev_enabled
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32


@contextmanager
def fallback_guard(enabled, x):
    if enabled and x.is_cuda and x.dtype == torch.float16:
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(
                enabled=False,
                benchmark=False,
                deterministic=True,
                allow_tf32=False,
            ):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
    else:
        yield


def tensor_diff(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def print_diff(name, a, b):
    dmax, dmean = tensor_diff(a, b)
    print(f"  {name:18s} max={dmax:.4e}  mean={dmean:.4e}")
    return dmax


def np_image_to_tensor(img, device, dtype):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image.to(dtype=dtype)


def read_frames(path, width, height, frame_num, device, dtype):
    reader = YUV420Reader(path, width, height)
    frames = []
    try:
        for _ in range(frame_num):
            y, uv = reader.read_one_frame()
            yuv = ycbcr420_to_444_np(y, uv)
            frames.append(np_image_to_tensor(yuv, device, dtype))
    finally:
        reader.close()
    return frames


def materialise_feature(model, ref_frame, use_guard):
    if ref_frame.feature is not None:
        with fallback_guard(use_guard, ref_frame.feature):
            return model.feature_adaptor_p(ref_frame.feature)
    ref = F.pixel_unshuffle(ref_frame.frame, 8)
    with fallback_guard(use_guard, ref):
        return model.feature_adaptor_i(ref)


def run_compress(model, x, qp, ref_frame, use_guard):
    model.clear_dpb()
    model.add_ref_frame(ref_frame.feature, ref_frame.frame)
    with fallback_guard(use_guard, x):
        encoded = model.compress(x, qp)
    next_ref = RefFrame()
    next_ref.frame = None
    next_ref.feature = model.dpb[0].feature
    return encoded, next_ref


def run_compress_batch(model, x_batch, qp, ref_frames, use_guard):
    ref_features = torch.cat(
        [materialise_feature(model, ref_frame, use_guard) for ref_frame in ref_frames],
        dim=0,
    )
    with fallback_guard(use_guard, x_batch):
        streams, features_out, _timing = model.compress_batch(x_batch, qp, ref_features)

    next_refs = []
    for idx in range(x_batch.shape[0]):
        ref = RefFrame()
        ref.frame = None
        ref.feature = features_out[idx:idx+1]
        next_refs.append(ref)
    return streams, next_refs, ref_features


def feature_extractor_trace(model, feature, qp, use_guard):
    q_feature = model.q_feature[qp:qp+1, :, :, :]
    with fallback_guard(use_guard, feature):
        fe_conv1_0 = model.feature_extractor.conv1[0](feature)
        fe_conv1_1 = model.feature_extractor.conv1[1](fe_conv1_0)
        ctx_t = fe_conv1_1 * q_feature
        fe_conv2_0 = model.feature_extractor.conv2[0](fe_conv1_1)
        fe_conv2_1 = model.feature_extractor.conv2[1](fe_conv2_0)
        fe_conv2_2 = model.feature_extractor.conv2[2](fe_conv2_1)
        fe_conv2_3 = model.feature_extractor.conv2[3](fe_conv2_2)
    return {
        "input": feature,
        "fe_conv1_0": fe_conv1_0,
        "fe_conv1_1": fe_conv1_1,
        "ctx_t": ctx_t,
        "fe_conv2_0": fe_conv2_0,
        "fe_conv2_1": fe_conv2_1,
        "fe_conv2_2": fe_conv2_2,
        "fe_conv2_3": fe_conv2_3,
        "ctx": fe_conv2_3,
    }


def clone_ref_from_iframe(x_hat):
    ref = RefFrame()
    ref.frame = x_hat
    ref.feature = None
    return ref


def init_model_pair(model_path_i, model_path_p, device, dtype, use_two_ec):
    i_net = DMCI()
    i_net.load_state_dict(get_state_dict(model_path_i))
    i_net = i_net.to(device).eval()
    if dtype == torch.float16:
        i_net.half()
    i_net.update()
    i_net.set_use_two_entropy_coders(use_two_ec)

    p_net = DMC()
    p_net.load_state_dict(get_state_dict(model_path_p))
    p_net = p_net.to(device).eval()
    if dtype == torch.float16:
        p_net.half()
    p_net.update()
    p_net.set_use_two_entropy_coders(use_two_ec)
    return i_net, p_net


def print_verdict(label, ok, success_text, fail_text):
    verdict = success_text if ok else fail_text
    print(f"[{label}] {verdict}")


def run_feature_extractor_level(args):
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not str(device).startswith("cuda"):
        raise RuntimeError("feature-extractor level requires CUDA to reproduce the bug")

    with cudnn_runtime(args.cudnn == "on"):
        i_net, p_net = init_model_pair(
            args.model_path_i,
            args.model_path_p,
            device,
            dtype,
            use_two_ec=False,
        )
        frames = read_frames(args.test_video, args.width, args.height, 1, device, dtype)
        i_encoded = i_net.compress(frames[0], args.qp)
        ref = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_feature = materialise_feature(p_net, ref, args.guard == "on")
        single_trace = feature_extractor_trace(p_net, ref_feature[0:1].contiguous(), args.qp, args.guard == "on")
        pair_input = torch.cat([ref_feature[0:1], ref_feature[0:1]], dim=0)
        pair_trace = feature_extractor_trace(p_net, pair_input, args.qp, args.guard == "on")

        print("=== feature-extractor ===")
        first_diff = None
        for name in (
            "input",
            "fe_conv1_0",
            "fe_conv1_1",
            "ctx_t",
            "fe_conv2_0",
            "fe_conv2_1",
            "fe_conv2_2",
            "fe_conv2_3",
            "ctx",
        ):
            dmax = print_diff(name, single_trace[name], pair_trace[name][0:1])
            if first_diff is None and dmax > args.threshold:
                first_diff = name
        if first_diff is None:
            print("  first_diff          none above threshold")
        else:
            print(f"  first_diff          {first_diff}")
        print_verdict(
            "feature-extractor",
            first_diff is None,
            "SUPPRESSED BY GUARD / RUNTIME",
            "REPRODUCED",
        )


def run_encode_level(args):
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not str(device).startswith("cuda"):
        raise RuntimeError("encode level requires CUDA to reproduce the bug")

    use_two_ec = (args.height * args.width) > (1280 * 720)
    with cudnn_runtime(args.cudnn == "on"):
        i_net, p_seq = init_model_pair(args.model_path_i, args.model_path_p, device, dtype, use_two_ec)
        _, p_b1 = init_model_pair(args.model_path_i, args.model_path_p, device, dtype, use_two_ec)
        _, p_b2 = init_model_pair(args.model_path_i, args.model_path_p, device, dtype, use_two_ec)

        frames = read_frames(args.test_video, args.width, args.height, 2, device, dtype)
        i_encoded = i_net.compress(frames[0], args.qp)
        ref_seq = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_b1 = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_b2_0 = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_b2_1 = clone_ref_from_iframe(i_encoded["x_hat"])

        mat_seq = materialise_feature(p_seq, ref_seq, args.guard == "on")
        mat_b1 = materialise_feature(p_b1, ref_b1, args.guard == "on")
        mat_b2_0 = materialise_feature(p_b2, ref_b2_0, args.guard == "on")
        mat_b2_1 = materialise_feature(p_b2, ref_b2_1, args.guard == "on")

        seq_encoded, ref_seq = run_compress(p_seq, frames[1], args.qp, ref_seq, args.guard == "on")
        b1_streams, b1_refs, _ = run_compress_batch(
            p_b1,
            frames[1],
            args.qp,
            [ref_b1],
            args.guard == "on",
        )
        ref_b1 = b1_refs[0]
        b2_streams, b2_refs, _ = run_compress_batch(
            p_b2,
            torch.cat([frames[1], frames[1]], dim=0),
            args.qp,
            [ref_b2_0, ref_b2_1],
            args.guard == "on",
        )
        ref_b2_0, ref_b2_1 = b2_refs

        print("=== encode ===")
        print_diff("mat_seq_vs_b1", mat_seq, mat_b1)
        print_diff("mat_seq_vs_b2", mat_seq, mat_b2_0)
        print_diff("b2_slot_internal", mat_b2_0, mat_b2_1)
        print_diff("feat_seq_vs_b1", ref_seq.feature, ref_b1.feature)
        print_diff("feat_seq_vs_b2", ref_seq.feature, ref_b2_0.feature)
        print_diff("b2_feat_internal", ref_b2_0.feature, ref_b2_1.feature)
        bits_b1 = seq_encoded["bit_stream"] == b1_streams[0]
        bits_b2 = seq_encoded["bit_stream"] == b2_streams[0]
        print(f"  bits_seq_vs_b1      {'yes' if bits_b1 else 'NO'}")
        print(f"  bits_seq_vs_b2      {'yes' if bits_b2 else 'NO'}")
        ok = bits_b1 and bits_b2 and tensor_diff(ref_seq.feature, ref_b2_0.feature)[0] <= args.threshold
        print_verdict("encode", ok, "SUPPRESSED BY GUARD / RUNTIME", "REPRODUCED")


def run_full_level(args):
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not str(device).startswith("cuda"):
        raise RuntimeError("full level requires CUDA to reproduce the bug")

    use_two_ec = (args.height * args.width) > (1280 * 720)
    with cudnn_runtime(args.cudnn == "on"):
        i_net, p_seq = init_model_pair(args.model_path_i, args.model_path_p, device, dtype, use_two_ec)
        _, p_b1 = init_model_pair(args.model_path_i, args.model_path_p, device, dtype, use_two_ec)
        _, p_b2 = init_model_pair(args.model_path_i, args.model_path_p, device, dtype, use_two_ec)

        frames = read_frames(args.test_video, args.width, args.height, args.num_frames, device, dtype)
        i_encoded = i_net.compress(frames[0], args.qp)
        ref_seq = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_b1 = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_b2_0 = clone_ref_from_iframe(i_encoded["x_hat"])
        ref_b2_1 = clone_ref_from_iframe(i_encoded["x_hat"])

        print("=== full ===")
        first_fail = None
        for frame_idx in range(1, len(frames)):
            mat_seq = materialise_feature(p_seq, ref_seq, args.guard == "on")
            mat_b1 = materialise_feature(p_b1, ref_b1, args.guard == "on")
            mat_b2_0 = materialise_feature(p_b2, ref_b2_0, args.guard == "on")

            seq_encoded, ref_seq = run_compress(p_seq, frames[frame_idx], args.qp, ref_seq, args.guard == "on")
            b1_streams, b1_refs, _ = run_compress_batch(
                p_b1, frames[frame_idx], args.qp, [ref_b1], args.guard == "on"
            )
            ref_b1 = b1_refs[0]
            b2_streams, b2_refs, _ = run_compress_batch(
                p_b2,
                torch.cat([frames[frame_idx], frames[frame_idx]], dim=0),
                args.qp,
                [ref_b2_0, ref_b2_1],
                args.guard == "on",
            )
            ref_b2_0, ref_b2_1 = b2_refs

            mat_b1_diff = tensor_diff(mat_seq, mat_b1)[0]
            mat_b2_diff = tensor_diff(mat_seq, mat_b2_0)[0]
            feat_b1_diff = tensor_diff(ref_seq.feature, ref_b1.feature)[0]
            feat_b2_diff = tensor_diff(ref_seq.feature, ref_b2_0.feature)[0]
            bits_b1 = seq_encoded["bit_stream"] == b1_streams[0]
            bits_b2 = seq_encoded["bit_stream"] == b2_streams[0]
            print(
                f"  frame {frame_idx:3d}  "
                f"mat_b1={mat_b1_diff:.4e}  mat_b2={mat_b2_diff:.4e}  "
                f"bits_b1={'yes' if bits_b1 else 'NO'}  bits_b2={'yes' if bits_b2 else 'NO'}  "
                f"feat_b1={feat_b1_diff:.4e}  feat_b2={feat_b2_diff:.4e}"
            )

            if first_fail is None:
                if mat_b1_diff > args.threshold or mat_b2_diff > args.threshold or \
                        feat_b1_diff > args.threshold or feat_b2_diff > args.threshold or \
                        (not bits_b1) or (not bits_b2):
                    first_fail = frame_idx

        print_verdict("full", first_fail is None, "SUPPRESSED BY GUARD / RUNTIME", "REPRODUCED")
        if first_fail is None:
            print("  first_fail          none")
        else:
            print(f"  first_fail          frame {first_fail}")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal repro ladder for batched fp16/cuDNN drift")
    parser.add_argument("--level", choices=("all", "full", "encode", "feature-extractor"), default="all")
    parser.add_argument("--guard", choices=("on", "off"), default="off")
    parser.add_argument("--cudnn", choices=("on", "off"), default="on")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--model_path_i", type=str, required=True)
    parser.add_argument("--model_path_p", type=str, required=True)
    parser.add_argument("--test_video", type=str, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_env()
    print(f"baseline_commit           5810653")
    print(f"level                     {args.level}")
    print(f"guard                     {args.guard}")
    print(f"cudnn                     {args.cudnn}")
    print(f"dtype                     {args.dtype}")
    print(f"video                     {args.test_video} ({args.width}x{args.height})")
    print()

    if args.level in ("all", "full"):
        run_full_level(args)
        print()
    if args.level in ("all", "encode"):
        run_encode_level(args)
        print()
    if args.level in ("all", "feature-extractor"):
        run_feature_extractor_level(args)


if __name__ == "__main__":
    main()

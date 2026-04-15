#!/usr/bin/env python3
"""
debug_compare_encoding.py

Comprehensive diagnostic that encodes a full video sequence using three
parallel paths and compares their outputs at every frame to identify
exactly where and why the batched path diverges from the sequential baseline.

THREE PATHS COMPARED
--------------------
1. SEQ     : The reference path. Uses p_frame_net.compress() exactly as
             test_video.py does — one frame at a time, model manages its
             own DPB internally.

2. BATCH-1 : Uses compress_batch() with B=1 (single sequence, no
             inter-sequence interaction). If this diverges from SEQ, the
             bug is inside compress_batch() itself — independent of any
             batching interaction between sequences.

3. BATCH-2 : Uses compress_batch() with B=2 (same frame fed as both
             sequences). If this diverges from BATCH-1, the bug is caused
             by inter-sequence state leakage when N>1.

WHAT WE MEASURE AT EACH FRAME
-------------------------------
For each P-frame we check four things:

  a) Materialised feature diff (input to the forward pass)
     The reference feature fed into the encoder. If this differs between
     paths at frame K, the state carried forward from frame K-1 is wrong —
     meaning ref_frames are being corrupted between frames.

  b) Bitstream match
     The actual encoded bytes. If these differ despite identical inputs,
     the entropy coder has stale internal state from a previous call.

  c) Decoder output feature diff (output of forward pass, stored as ref
     for next frame). If this differs despite identical inputs, there is a
     numerical divergence inside compress_batch() vs compress().

  d) Cumulative drift
     The running max absolute difference between SEQ and BATCH-1/BATCH-2
     reference features, accumulated across all frames. This detects small
     per-frame errors that would be invisible to the per-frame threshold
     but compound into visible PSNR loss over a full sequence.

  e) Internal BATCH-2 consistency (slot 0 vs slot 1)
     Since both B=2 slots receive identical inputs, their outputs must be
     identical. Any divergence here means the B=2 forward pass is
     corrupting one slot with the other's state.

AUTO-DIAGNOSIS LOGIC
--------------------
After the full run, the script prints a diagnosis based on which
divergence trackers fired, checked independently (not in an elif chain):

  - If BATCH-1 decoder output diverges from SEQ:
    => Bug is inside compress_batch() itself (B=1 already breaks it).

  - If BATCH-1 matches SEQ but BATCH-2 diverges:
    => Bug is inter-sequence state leakage in the B=2 forward pass.

  - If materialised features diverge before decoder outputs:
    => Bug is in how ref_frames state is stored/restored between frames.

  - If only bitstreams diverge (features match):
    => Bug is entropy coder state leaking between calls.

  - If cumulative drift is significant but per-frame diffs are small:
    => Small per-frame errors are compounding — check float16 precision.

  - If nothing diverges:
    => Encoding is correct. Bug is in the decode path (not tested here).
    => Re-run with --also_decode to test the full encode+decode cycle.

HOW TO RUN
----------
    python debug_compare_encoding.py \\
        --model_path_i ./checkpoints/cvpr2025_image.pth.tar \\
        --model_path_p ./checkpoints/cvpr2025_video.pth.tar \\
        --test_video  test_data/UVG/ShakeNDry_640x360_120fps_420_8bit_YUV.yuv \\
        --width 640 --height 360 --qp 0 --num_frames 100 \\
        --reset_interval 32

IMPORTANT: --reset_interval must match the value used in test_video_batch.py
           (default 32, same as this script's default).
"""

import argparse
import copy
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from src.models.image_model import DMCI
from src.models.video_model import DMC, RefFrame
from src.layers.cuda_inference import round_and_to_int8
from src.utils.common import get_state_dict, set_torch_env
from src.utils.transforms import ycbcr420_to_444_np
from src.utils.video_reader import YUV420Reader


# ---------------------------------------------------------------------------
# Helpers — mirror test_video_batch.py exactly
# ---------------------------------------------------------------------------

def replicate_pad(x, pad_b, pad_r):
    if pad_b > 0 or pad_r > 0:
        return F.pad(x, (0, pad_r, 0, pad_b), mode='replicate')
    return x


def materialise_feature(p_frame_net, ref_frame):
    """
    Derive the reference feature that the encoder/decoder will use as
    context for this frame.  Mirrors DMC.apply_feature_adaptor() exactly:
      - After I-frame  : pixel_unshuffle the reconstructed pixels → feature_adaptor_i
      - After P-frame  : feature_adaptor_p on the stored decoder output feature
    We call this externally (without touching self.dpb) so the same tensor
    can be reused across all three paths without the model's DPB getting in the way.
    """
    if ref_frame.feature is not None:
        return p_frame_net.feature_adaptor_p(ref_frame.feature)
    return p_frame_net.feature_adaptor_i(F.pixel_unshuffle(ref_frame.frame, 8))


def tensor_diff_f32(a, b):
    """
    Compute max and mean absolute difference in float32.
    We explicitly upcast from float16 here because float16 has a machine
    epsilon of ~9.77e-4 near 1.0 — differences smaller than that are
    invisible if we compare in half precision.  Upcasting to float32
    (epsilon ~1.2e-7) lets us detect much smaller discrepancies.
    """
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def print_named_diff(name, a, b, indent="  "):
    dmax, dmean = tensor_diff_f32(a, b)
    print(f"{indent}{name:18s} max={dmax:.4e}  mean={dmean:.4e}")
    return dmax, dmean


def bytes_match(a, b):
    return a == b


def compress_prior_2x_trace(model, y, common_params):
    """
    Mirror CompressionModel.compress_prior_2x() but keep all relevant
    intermediates so we can identify the first divergence inside the prior path.
    """
    y_scaled, q_dec, scales_0, means_0 = \
        model.separate_prior_for_video_encoding(common_params.clone(), y.clone())
    dtype = y_scaled.dtype
    device = y_scaled.device
    B, C, H, W = y_scaled.size()
    mask_0, mask_1 = model.get_mask_2x(B, C, H, W, dtype, device)

    _, y_q_0, y_hat_0, s_hat_0 = model.process_with_mask(y_scaled, scales_0, means_0, mask_0)
    cat_params = torch.cat((y_hat_0, common_params), dim=1)
    scales_1, means_1 = model.y_spatial_prior(cat_params).chunk(2, 1)
    _, y_q_1, y_hat_1, s_hat_1 = model.process_with_mask(y_scaled, scales_1, means_1, mask_1)
    y_hat = (y_hat_0 + y_hat_1) * q_dec

    return {
        'prior_y_scaled': y_scaled,
        'prior_q_dec': q_dec,
        'prior_scales_0': scales_0,
        'prior_means_0': means_0,
        'prior_y_q_0': y_q_0,
        'prior_y_hat_0': y_hat_0,
        'prior_s_hat_0': s_hat_0,
        'prior_cat_params': cat_params,
        'prior_scales_1': scales_1,
        'prior_means_1': means_1,
        'prior_y_q_1': y_q_1,
        'prior_y_hat_1': y_hat_1,
        'prior_s_hat_1': s_hat_1,
        'prior_y_hat': y_hat,
    }


def forward_trace(model, x, qp, ref_feature):
    """
    Run the forward part of compress() without touching DPB or entropy coder.
    Returns intermediate tensors so we can compare SEQ/B1/B2 stage-by-stage.
    """
    q_encoder = model.q_encoder[qp:qp+1, :, :, :]
    q_decoder = model.q_decoder[qp:qp+1, :, :, :]
    q_feature = model.q_feature[qp:qp+1, :, :, :]

    with model._fallback_conv_guard(x):
        ctx, ctx_t = model.extract_context(ref_feature, q_feature)
        y = model.encoder(x, ctx, q_encoder)
        z = model.hyper_encoder(model.pad_for_y(y))
        z_hat, _ = round_and_to_int8(z)
        params = model.res_prior_param_decoder(z_hat, ctx_t)
        prior_trace = compress_prior_2x_trace(model, y, params)
        y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = \
            model.compress_prior_2x(y, params, model.y_spatial_prior)
        features_out = model.decoder(y_hat, ctx, q_decoder)

    trace = {
        'ctx': ctx,
        'ctx_t': ctx_t,
        'y': y,
        'z_hat': z_hat,
        'params': params,
        'y_q_w_0': y_q_w_0,
        'y_q_w_1': y_q_w_1,
        's_w_0': s_w_0,
        's_w_1': s_w_1,
        'y_hat': y_hat,
        'features_out': features_out,
    }
    trace.update(prior_trace)
    return trace


def feature_extractor_trace(feature_extractor, ref_feature, q_feature):
    """
    Trace FeatureExtractor block-by-block so we can see whether the first
    divergence appears in conv1[0], conv1[1], or only later in conv2.
    """
    block_10 = feature_extractor.conv1[0](ref_feature)
    block_11 = feature_extractor.conv1[1](block_10)
    ctx_t = block_11 * q_feature

    block_20 = feature_extractor.conv2[0](block_11)
    block_21 = feature_extractor.conv2[1](block_20)
    block_22 = feature_extractor.conv2[2](block_21)
    block_23 = feature_extractor.conv2[3](block_22)

    return {
        'ref_feature': ref_feature,
        'fe_conv1_0': block_10,
        'fe_conv1_1': block_11,
        'ctx_t': ctx_t,
        'fe_conv2_0': block_20,
        'fe_conv2_1': block_21,
        'fe_conv2_2': block_22,
        'fe_conv2_3': block_23,
        'ctx': block_23,
    }


def print_feature_extractor_ablation(model, ref_feature, qp):
    """
    Compare FeatureExtractor(B=1) against FeatureExtractor(B=2) where the same
    sample is repeated twice. Then rerun the same test with cuDNN disabled and
    with the subgraph upcast to float32.
    """
    single_ref = ref_feature[0:1].contiguous()
    pair_ref = torch.cat([single_ref, single_ref], dim=0)
    q_feature = model.q_feature[qp:qp+1, :, :, :]

    def run_trace(feature_extractor, one_ref, two_ref, qf):
        trace_one = feature_extractor_trace(feature_extractor, one_ref, qf)
        trace_two = feature_extractor_trace(feature_extractor, two_ref, qf)
        return trace_one, slice_trace(trace_two, 0)

    def print_ablation(label, trace_one, trace_two):
        print(f"[ABLATION] {label}")
        first_diff = None
        for name in (
            'ref_feature',
            'fe_conv1_0',
            'fe_conv1_1',
            'ctx_t',
            'fe_conv2_0',
            'fe_conv2_1',
            'fe_conv2_2',
            'fe_conv2_3',
            'ctx',
        ):
            dmax, _ = print_named_diff(name, trace_one[name], trace_two[name])
            if first_diff is None and dmax > 1e-3:
                first_diff = name
        if first_diff is None:
            print("  first_diff          none above threshold")
        else:
            print(f"  first_diff          {first_diff}")

    trace_one, trace_two = run_trace(model.feature_extractor, single_ref, pair_ref, q_feature)
    print_ablation("feature_extractor half, current backend", trace_one, trace_two)

    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        trace_one, trace_two = run_trace(model.feature_extractor, single_ref, pair_ref, q_feature)
    finally:
        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    print_ablation("feature_extractor half, cuDNN disabled", trace_one, trace_two)

    feature_extractor_fp32 = copy.deepcopy(model.feature_extractor).float().to(single_ref.device).eval()
    single_ref_fp32 = single_ref.float()
    pair_ref_fp32 = pair_ref.float()
    q_feature_fp32 = q_feature.float()
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        trace_one, trace_two = run_trace(
            feature_extractor_fp32,
            single_ref_fp32,
            pair_ref_fp32,
            q_feature_fp32,
        )
    finally:
        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    print_ablation("feature_extractor float32", trace_one, trace_two)


def slice_trace(trace, idx):
    return {k: v[idx:idx+1] for k, v in trace.items()}


def print_stage_diffs(label_a, trace_a, label_b, trace_b):
    print(f"[TRACE] {label_a} vs {label_b}")
    first_diff = None
    for name in (
        'ctx', 'ctx_t', 'y', 'z_hat', 'params',
        'prior_y_scaled', 'prior_q_dec',
        'prior_scales_0', 'prior_means_0', 'prior_y_q_0', 'prior_y_hat_0', 'prior_s_hat_0',
        'prior_cat_params',
        'prior_scales_1', 'prior_means_1', 'prior_y_q_1', 'prior_y_hat_1', 'prior_s_hat_1',
        'prior_y_hat',
        'y_q_w_0', 'y_q_w_1', 's_w_0', 's_w_1', 'y_hat', 'features_out'
    ):
        dmax, dmean = tensor_diff_f32(trace_a[name], trace_b[name])
        print(f"  {name:14s} max={dmax:.4e}  mean={dmean:.4e}")
        if first_diff is None and dmax > 1e-3:
            first_diff = name
    if first_diff is None:
        print("  first_diff      none above threshold")
    else:
        print(f"  first_diff      {first_diff}")


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential vs batched encoding frame-by-frame")
    parser.add_argument('--model_path_i',    type=str, required=True)
    parser.add_argument('--model_path_p',    type=str, required=True)
    parser.add_argument('--test_video',      type=str, required=True)
    parser.add_argument('--width',           type=int, default=640)
    parser.add_argument('--height',          type=int, default=360)
    parser.add_argument('--qp',              type=int, default=0)
    parser.add_argument('--num_frames',      type=int, default=100)
    parser.add_argument('--reset_interval',  type=int, default=32,
                        help="Must match --reset_interval used in test_video_batch.py (default: 32)")
    parser.add_argument('--trace_frame',     type=int, default=1,
                        help="P-frame index at which to print detailed SEQ/B1/B2 stage diffs")
    args = parser.parse_args()

    set_torch_env()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # -----------------------------------------------------------------------
    # Determine two-entropy-coder mode — mirrors test_video_batch.py line 488
    # use_two_entropy_coders is True for content larger than 1280x720.
    # This flag changes the internal RANS codec configuration and must be
    # set before any encoding/decoding calls, otherwise the bitstreams will
    # be structured differently from what the test harness produces.
    # -----------------------------------------------------------------------
    use_two_ec = (args.height * args.width) > (1280 * 720)

    print(f"Device:                  {device}")
    print(f"Video:                   {args.test_video}  ({args.width}x{args.height})")
    print(f"QP:                      {args.qp}")
    print(f"Frames:                  {args.num_frames}")
    print(f"Reset interval:          {args.reset_interval}")
    print(f"use_two_entropy_coders:  {use_two_ec}")
    print()

    # -----------------------------------------------------------------------
    # Load models — mirrors test_video_batch.py init_func()
    # -----------------------------------------------------------------------
    i_frame_net = DMCI()
    i_frame_net.load_state_dict(get_state_dict(args.model_path_i))
    i_frame_net = i_frame_net.to(device).eval()
    i_frame_net.half()
    i_frame_net.update()
    i_frame_net.set_use_two_entropy_coders(use_two_ec)

    p_frame_net = DMC()
    p_frame_net.load_state_dict(get_state_dict(args.model_path_p))
    p_frame_net = p_frame_net.to(device).eval()
    p_frame_net.half()
    p_frame_net.update()
    p_frame_net.set_use_two_entropy_coders(use_two_ec)

    w, h = args.width, args.height
    pad_r, pad_b = DMCI.get_padding_size(h, w, 16)
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    # -----------------------------------------------------------------------
    # Read all frames up front
    # -----------------------------------------------------------------------
    reader = YUV420Reader(args.test_video, w, h)
    frames = []
    for _ in range(args.num_frames):
        y, uv = reader.read_one_frame()
        yuv = ycbcr420_to_444_np(y, uv)
        x = torch.from_numpy(yuv).to(device=device, dtype=torch.float16).unsqueeze(0) / 255.0
        frames.append(replicate_pad(x, pad_b, pad_r))
    reader.close()
    print(f"Read {len(frames)} frames.\n")

    # -----------------------------------------------------------------------
    # Per-path ref state — one RefFrame per path
    # -----------------------------------------------------------------------
    ref_seq  = RefFrame()   # SEQ path
    ref_b1   = RefFrame()   # BATCH-1 path
    ref_b2_0 = RefFrame()   # BATCH-2 slot 0
    ref_b2_1 = RefFrame()   # BATCH-2 slot 1 (receives identical input to slot 0)

    last_qp_seq = last_qp_b1 = last_qp_b2 = 0

    # -----------------------------------------------------------------------
    # Divergence trackers
    # These track the *first* frame at which each type of divergence occurs.
    # Checked independently (not in an elif chain) so multiple root causes
    # can be identified simultaneously.
    # -----------------------------------------------------------------------
    first_mat_div_b1   = None   # materialised feature: BATCH-1 vs SEQ
    first_mat_div_b2   = None   # materialised feature: BATCH-2 vs SEQ
    first_bits_div_b1  = None   # bitstream bytes:      BATCH-1 vs SEQ
    first_bits_div_b2  = None   # bitstream bytes:      BATCH-2 vs SEQ
    first_feat_div_b1  = None   # decoder output feat:  BATCH-1 vs SEQ
    first_feat_div_b2  = None   # decoder output feat:  BATCH-2 vs SEQ
    first_b2_internal  = None   # BATCH-2 internal:     slot 0 vs slot 1

    # -----------------------------------------------------------------------
    # Cumulative drift accumulators
    # We add the per-frame max absolute feature difference to these running
    # totals to detect small-but-compounding errors that would be invisible
    # if you only look at the per-frame threshold.
    # -----------------------------------------------------------------------
    cum_drift_b1 = 0.0   # sum of per-frame max|feat_b1 - feat_seq|
    cum_drift_b2 = 0.0   # sum of per-frame max|feat_b2 - feat_seq|

    # Threshold for flagging a per-frame difference as significant.
    # Using 1e-3 (relative to typical feature magnitudes) rather than 1e-4
    # because float16 machine epsilon near 1.0 is ~9.77e-4, so anything
    # below that threshold could simply be float16 rounding.
    THRESH = 1e-3

    # -----------------------------------------------------------------------
    # Per-frame loop
    # -----------------------------------------------------------------------
    print("=" * 110)
    print(f"{'Frame':>5}  {'T':>1}  "
          f"{'MatDiff B1-SEQ':>16}  {'MatDiff B2-SEQ':>16}  "
          f"{'Bits B1':>7}  {'Bits B2':>7}  "
          f"{'FeatDiff B1-SEQ':>16}  {'FeatDiff B2-SEQ':>16}  "
          f"{'B2 internal':>12}  {'Notes'}")
    print("=" * 110)

    with torch.no_grad():
        for frame_idx in range(args.num_frames):
            x    = frames[frame_idx]
            qp_i = args.qp
            fa_idx = index_map[frame_idx % 8]
            qp_p   = p_frame_net.shift_qp(args.qp, fa_idx)
            is_i   = (frame_idx == 0)

            # ------------------------------------------------------------------
            # I-FRAME (frame 0): all paths use the same i_frame_net.compress().
            # All paths start with the same reconstructed I-frame x_hat,
            # so their ref states are initialised identically.
            # ------------------------------------------------------------------
            if is_i:
                encoded_i = i_frame_net.compress(x, qp_i)
                x_hat_i   = encoded_i['x_hat']

                for ref in (ref_seq, ref_b1, ref_b2_0, ref_b2_1):
                    ref.frame   = x_hat_i
                    ref.feature = None

                last_qp_seq = last_qp_b1 = last_qp_b2 = qp_i
                print(f"{frame_idx:>5}  I  {'(shared I-frame)':>16}  {'(shared I-frame)':>16}  "
                      f"{'yes':>7}  {'yes':>7}  "
                      f"{'(shared I-frame)':>16}  {'(shared I-frame)':>16}  "
                      f"{'ok':>12}")
                continue

            # ------------------------------------------------------------------
            # RESET INTERVAL — mirrors test_video_batch.py lines 291-299.
            # At frame_idx % reset_interval == 1, prepare_feature_adaptor_i()
            # regenerates the pixel-space reference from the stored feature.
            # We apply this to all four ref states using the same code path
            # as the test harness, so all paths remain in sync.
            # Note: this is a decoder-side preparation step; it sets
            # dpb[0].feature = None so that apply_feature_adaptor() will
            # use the pixel path (feature_adaptor_i) on the next frame.
            # ------------------------------------------------------------------
            if args.reset_interval > 0 and frame_idx % args.reset_interval == 1:
                for ref, lqp in [(ref_seq,  last_qp_seq),
                                  (ref_b1,   last_qp_b1),
                                  (ref_b2_0, last_qp_b2),
                                  (ref_b2_1, last_qp_b2)]:
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(ref.feature, ref.frame)
                    p_frame_net.prepare_feature_adaptor_i(lqp)
                    ref.frame   = p_frame_net.dpb[0].frame
                    ref.feature = p_frame_net.dpb[0].feature

            # ------------------------------------------------------------------
            # MATERIALISE FEATURES
            # Compute the reference feature that the encoder will use as context.
            # We do this BEFORE the forward passes so we can compare the inputs
            # independently of the outputs.  If these differ, the bug is in
            # how ref_frames are stored/restored between frames — not inside
            # the forward pass itself.
            # All four materialise calls use the same stateless neural layers
            # (feature_adaptor_p / feature_adaptor_i), so running them
            # sequentially does not cause state contamination.
            # ------------------------------------------------------------------
            mat_seq  = materialise_feature(p_frame_net, ref_seq)
            mat_b1   = materialise_feature(p_frame_net, ref_b1)
            mat_b2_0 = materialise_feature(p_frame_net, ref_b2_0)
            mat_b2_1 = materialise_feature(p_frame_net, ref_b2_1)

            # Compare in float32 to avoid float16 epsilon masking differences
            mat_diff_b1, _ = tensor_diff_f32(mat_b1,   mat_seq)
            mat_diff_b2, _ = tensor_diff_f32(mat_b2_0, mat_seq)

            if first_mat_div_b1 is None and mat_diff_b1 > THRESH:
                first_mat_div_b1 = frame_idx
            if first_mat_div_b2 is None and mat_diff_b2 > THRESH:
                first_mat_div_b2 = frame_idx

            # ------------------------------------------------------------------
            # PATH 1: SEQUENTIAL — p_frame_net.compress() with internal DPB.
            # We clear and reload the DPB with ref_seq state before each call
            # so the model's DPB exactly matches what this path expects.
            # compress() will call self.apply_feature_adaptor() → reads dpb[0],
            # then at the end calls self.add_ref_frame() → writes dpb[0].
            # We read the decoder output feature back from dpb[0] after the call.
            # ------------------------------------------------------------------
            p_frame_net.clear_dpb()
            p_frame_net.add_ref_frame(ref_seq.feature, ref_seq.frame)
            enc_seq  = p_frame_net.compress(x, qp_p)
            bits_seq = enc_seq['bit_stream']
            feat_seq = p_frame_net.dpb[0].feature.clone()

            ref_seq.feature = feat_seq
            ref_seq.frame   = None
            last_qp_seq     = qp_p

            # ------------------------------------------------------------------
            # PATH 2: BATCH-1 — compress_batch() with B=1.
            # compress_batch() does NOT read from or write to self.dpb — it
            # receives the reference feature as an explicit argument (mat_b1)
            # and returns decoder output features directly.  This means it is
            # fully stateless w.r.t. the model's DPB.
            # If B=1 already diverges from SEQ despite identical inputs,
            # the bug is inside the compress_batch() implementation itself.
            # ------------------------------------------------------------------
            streams_b1, feats_b1, _ = p_frame_net.compress_batch(x, qp_p, mat_b1)
            bits_b1  = streams_b1[0]
            feat_b1  = feats_b1[0:1].clone()

            ref_b1.feature = feat_b1
            ref_b1.frame   = None
            last_qp_b1     = qp_p

            # ------------------------------------------------------------------
            # PATH 3: BATCH-2 — compress_batch() with B=2 (same frame twice).
            # Both slots receive identical input tensors and identical reference
            # features.  Their outputs MUST be identical (slot 0 == slot 1).
            # Any divergence between slots exposes inter-sequence state leakage
            # inside the B=2 forward pass.
            # Any divergence vs BATCH-1 (which ran the same computation at B=1)
            # means something about the B=2 batching itself changes the result.
            # ------------------------------------------------------------------
            x_batch  = torch.cat([x, x],             dim=0)
            mat_b2   = torch.cat([mat_b2_0, mat_b2_1], dim=0)
            p_frame_net._debug_batch_diff = (frame_idx == 1)  # print only on first P-frame
            streams_b2, feats_b2, _ = p_frame_net.compress_batch(x_batch, qp_p, mat_b2)
            p_frame_net._debug_batch_diff = False
            bits_b2_0 = streams_b2[0]
            bits_b2_1 = streams_b2[1]
            feat_b2_0 = feats_b2[0:1].clone()
            feat_b2_1 = feats_b2[1:2].clone()

            ref_b2_0.feature = feat_b2_0
            ref_b2_0.frame   = None
            ref_b2_1.feature = feat_b2_1
            ref_b2_1.frame   = None
            last_qp_b2       = qp_p

            if frame_idx == args.trace_frame:
                print("[TRACE] materialised reference feature inputs")
                print_named_diff("SEQ vs B1", mat_seq, mat_b1)
                print_named_diff("SEQ vs B2[0]", mat_seq, mat_b2_0)
                print_named_diff("B2[0] vs B2[1]", mat_b2_0, mat_b2_1)
                trace_seq = forward_trace(p_frame_net, x, qp_p, mat_seq)
                trace_b1 = forward_trace(p_frame_net, x, qp_p, mat_b1)
                trace_b2 = forward_trace(p_frame_net, x_batch, qp_p, mat_b2)

                print_stage_diffs("SEQ", trace_seq, "B1", trace_b1)
                print_stage_diffs("SEQ", trace_seq, "B2[0]", slice_trace(trace_b2, 0))
                print_stage_diffs("B1", trace_b1, "B2[0]", slice_trace(trace_b2, 0))
                print_stage_diffs("B2[0]", slice_trace(trace_b2, 0),
                                  "B2[1]", slice_trace(trace_b2, 1))
                print_feature_extractor_ablation(p_frame_net, mat_seq, qp_p)

            # ------------------------------------------------------------------
            # COMPARE RESULTS
            # ------------------------------------------------------------------

            # Bitstreams (bytes-exact comparison)
            bm_b1  = bytes_match(bits_b1,   bits_seq)
            bm_b2  = bytes_match(bits_b2_0, bits_seq)
            bm_int = bytes_match(bits_b2_0, bits_b2_1)  # B=2 internal consistency

            if first_bits_div_b1 is None and not bm_b1:
                first_bits_div_b1 = frame_idx
            if first_bits_div_b2 is None and not bm_b2:
                first_bits_div_b2 = frame_idx

            # Decoder output features (in float32 for sensitivity)
            feat_diff_b1, feat_mean_b1 = tensor_diff_f32(feat_b1,   feat_seq)
            feat_diff_b2, feat_mean_b2 = tensor_diff_f32(feat_b2_0, feat_seq)
            feat_diff_int, _           = tensor_diff_f32(feat_b2_0, feat_b2_1)

            if first_feat_div_b1 is None and feat_diff_b1 > THRESH:
                first_feat_div_b1 = frame_idx
            if first_feat_div_b2 is None and feat_diff_b2 > THRESH:
                first_feat_div_b2 = frame_idx
            if first_b2_internal is None and feat_diff_int > THRESH:
                first_b2_internal = frame_idx

            # Cumulative drift — accumulated max difference in reference features.
            # Even if per-frame differences are below THRESH, a non-zero
            # accumulation here means errors are building up over time.
            cum_drift_b1 += feat_diff_b1
            cum_drift_b2 += feat_diff_b2

            # ------------------------------------------------------------------
            # Per-frame print
            # Flags: !! marks any column that exceeds THRESH or mismatches.
            # ------------------------------------------------------------------
            notes = []
            if mat_diff_b1 > THRESH:  notes.append("MAT-B1!")
            if mat_diff_b2 > THRESH:  notes.append("MAT-B2!")
            if not bm_b1:             notes.append("BITS-B1!")
            if not bm_b2:             notes.append("BITS-B2!")
            if not bm_int:            notes.append("BITS-INT!")
            if feat_diff_b1 > THRESH: notes.append("FEAT-B1!")
            if feat_diff_b2 > THRESH: notes.append("FEAT-B2!")
            if feat_diff_int > THRESH:notes.append("FEAT-INT!")

            b2int_str = "ok" if (bm_int and feat_diff_int <= THRESH) else f"{feat_diff_int:.2e}"

            print(f"{frame_idx:>5}  P  "
                  f"{mat_diff_b1:>16.4e}  {mat_diff_b2:>16.4e}  "
                  f"{'yes' if bm_b1 else 'NO':>7}  {'yes' if bm_b2 else 'NO':>7}  "
                  f"{feat_diff_b1:>16.4e}  {feat_diff_b2:>16.4e}  "
                  f"{b2int_str:>12}  "
                  f"{' '.join(notes)}")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"  First frame: materialised feature diverges  B1 vs SEQ : {first_mat_div_b1  or 'never'}")
    print(f"  First frame: materialised feature diverges  B2 vs SEQ : {first_mat_div_b2  or 'never'}")
    print(f"  First frame: bitstream diverges             B1 vs SEQ : {first_bits_div_b1 or 'never'}")
    print(f"  First frame: bitstream diverges             B2 vs SEQ : {first_bits_div_b2 or 'never'}")
    print(f"  First frame: decoder feature diverges       B1 vs SEQ : {first_feat_div_b1 or 'never'}")
    print(f"  First frame: decoder feature diverges       B2 vs SEQ : {first_feat_div_b2 or 'never'}")
    print(f"  First frame: BATCH-2 internal inconsistency (s0 vs s1): {first_b2_internal or 'never'}")
    print(f"  Cumulative feature drift  B1 vs SEQ (sum of per-frame max): {cum_drift_b1:.6f}")
    print(f"  Cumulative feature drift  B2 vs SEQ (sum of per-frame max): {cum_drift_b2:.6f}")
    print()

    # -----------------------------------------------------------------------
    # AUTO-DIAGNOSIS
    # Checks each failure mode independently.  Multiple may be printed if
    # multiple root causes are present simultaneously.
    # -----------------------------------------------------------------------
    print("DIAGNOSIS")
    print("-" * 110)
    diagnosed = False

    # Check B1 first — if B1 already breaks, B2 is irrelevant.
    if first_feat_div_b1 is not None:
        print(f"[B1 FAIL] compress_batch(B=1) produces different decoder output features")
        print(f"          from compress() starting at frame {first_feat_div_b1}, despite receiving")
        print(f"          identical input tensors.  The bug is INSIDE compress_batch() itself,")
        print(f"          independent of batching.  Compare the two implementations carefully:")
        print(f"          compress() in video_model.py:299 vs compress_batch() in video_model.py:343.")
        diagnosed = True

    if first_bits_div_b1 is not None and first_feat_div_b1 is None:
        print(f"[BITS-B1] Bitstreams diverge at frame {first_bits_div_b1} but decoder output features match.")
        print(f"          The forward pass is numerically identical but the entropy coder")
        print(f"          is producing different bytes.  Likely cause: entropy_coder has")
        print(f"          stale internal state from the previous compress() call that is")
        print(f"          not fully cleared by entropy_coder.reset() in compress_batch().")
        diagnosed = True

    # Check B2 independently of B1.
    if first_b2_internal is not None:
        print(f"[B2 INT]  BATCH-2 slots 0 and 1 diverge at frame {first_b2_internal} despite")
        print(f"          receiving IDENTICAL inputs.  This is direct evidence of inter-sequence")
        print(f"          state leakage within the B=2 forward pass in compress_batch().")
        diagnosed = True

    if first_feat_div_b2 is not None and first_feat_div_b1 is None:
        print(f"[B2 FAIL] BATCH-1 matches SEQ but BATCH-2 diverges at frame {first_feat_div_b2}.")
        print(f"          The bug is specifically caused by processing N=2 sequences together.")
        print(f"          Likely cause: shared mutable state (mask cache, entropy coder internal")
        print(f"          buffer, or CUDA stream) is being contaminated across the two slots.")
        diagnosed = True

    if first_bits_div_b2 is not None and first_feat_div_b2 is None and first_feat_div_b1 is None:
        print(f"[BITS-B2] BATCH-2 bitstreams diverge at frame {first_bits_div_b2} but features match.")
        print(f"          Same entropy coder state issue as BITS-B1 but only manifests at B=2.")
        diagnosed = True

    # Check materialised features — this fires if the ref_frames state is wrong.
    if first_mat_div_b1 is not None or first_mat_div_b2 is not None:
        b1_f = first_mat_div_b1 or 'never'
        b2_f = first_mat_div_b2 or 'never'
        print(f"[MAT DIV] Materialised features diverge (B1 at frame {b1_f}, B2 at frame {b2_f}).")
        print(f"          The reference features stored in ref_frames between frames are wrong.")
        print(f"          Check how ref_frames[i].feature is set after each compress_batch() call")
        print(f"          and compare with how compress() updates self.dpb.")
        diagnosed = True

    # Cumulative drift without per-frame threshold violations.
    if not diagnosed and (cum_drift_b1 > 0 or cum_drift_b2 > 0):
        print(f"[DRIFT]   No per-frame differences exceed the {THRESH:.0e} threshold, but")
        print(f"          cumulative drift is non-zero (B1={cum_drift_b1:.6f}, B2={cum_drift_b2:.6f}).")
        print(f"          Small float16 rounding differences are accumulating over time.")
        print(f"          This could explain gradual PSNR degradation.  Consider whether")
        print(f"          the half-precision arithmetic in compress_batch() matches compress().")
        diagnosed = True

    if not diagnosed:
        print("[PASS]    All encoding paths match perfectly across all frames.")
        print("          The PSNR bug is NOT in the encode path.")
        print("          Next step: investigate the DECODE path.")
        print("          The most likely candidate is decompress() reading stale DPB state,")
        print("          specifically the reset_ref_feature() interaction at reset-interval frames.")
        print("          The decoder in test_video_batch.py:decode_sequence() should be audited.")


if __name__ == "__main__":
    main()

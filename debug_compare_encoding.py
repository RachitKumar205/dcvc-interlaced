#!/usr/bin/env python3
"""
debug_compare_encoding.py

Comprehensive diagnostic that encodes a full video sequence using three
parallel paths and compares their outputs at every frame to identify
exactly where and why the batched path diverges from the sequential baseline.

THREE PATHS COMPARED
--------------------
1. SEQUENTIAL   : The reference path. Uses p_frame_net.compress() exactly
                  as test_video.py does — one frame at a time, model manages
                  its own DPB internally.

2. BATCH-1      : Uses compress_batch() with B=1 (single sequence, no
                  inter-sequence interaction). If this diverges from SEQUENTIAL,
                  the bug is inside compress_batch() itself — independent of
                  any batching interaction between sequences.

3. BATCH-2      : Uses compress_batch() with B=2 (same frame duplicated).
                  If this diverges from BATCH-1, the bug is caused by
                  inter-sequence state leakage when N>1.

WHAT WE CHECK AT EACH FRAME
----------------------------
For each P-frame:

  a) Materialised feature (input to forward pass)
     The reference feature passed into the encoder. If this differs between
     paths, the state carried from the previous frame is wrong — the bug is
     in how we store/restore ref_frames between frames.

  b) Bitstream bytes
     The actual encoded bytes. If these differ despite identical inputs,
     the entropy coder has state left over from a previous call.

  c) Decoder output feature (output of forward pass)
     The feature stored back into ref_frames for the next frame. If this
     differs despite identical inputs, there is a numerical issue inside
     compress_batch() vs compress().

  d) Cumulative drift
     We track the max absolute difference in the running ref_feature across
     all frames to detect if small per-frame errors compound over time.

HOW TO READ THE OUTPUT
----------------------
- If materialised features match but bitstreams differ: entropy coder state leak
- If materialised features differ: ref_frame state is being corrupted between frames
- If both materialised features AND bitstreams match but decoder output differs:
  bug is inside compress_batch() forward pass
- BATCH-1 == SEQ but BATCH-2 != SEQ: inter-sequence state leakage in B=2 path
- BATCH-1 != SEQ: bug is in compress_batch() itself, independent of batch size

Usage:
    python debug_compare_encoding.py \
        --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
        --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
        --test_video  test_data/UVG/ShakeNDry_640x360_120fps_420_8bit_YUV.yuv \
        --width 640 --height 360 --qp 0 --num_frames 100
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.models.common_model import RefFrame
from src.utils.common import get_state_dict, set_torch_env
from src.utils.yuv import YUV420Reader


# ---------------------------------------------------------------------------
# Helpers (mirrors test_video_batch.py exactly so we test the same logic)
# ---------------------------------------------------------------------------

def ycbcr420_to_444_np(y, uv):
    return np.stack([y, uv[0], uv[1]], axis=0)


def replicate_pad(x, pad_b, pad_r):
    if pad_b > 0 or pad_r > 0:
        return F.pad(x, (0, pad_r, 0, pad_b), mode='replicate')
    return x


def materialise_feature(p_frame_net, ref_frame):
    """
    Derive the reference feature for the encoder/decoder without modifying
    p_frame_net.dpb. Mirrors apply_feature_adaptor() in the model:
      - After I-frame: pixel_unshuffle the reconstructed pixels, then feature_adaptor_i
      - After P-frame: feature_adaptor_p on the stored decoder output feature
    """
    if ref_frame.feature is not None:
        return p_frame_net.feature_adaptor_p(ref_frame.feature)
    return p_frame_net.feature_adaptor_i(F.pixel_unshuffle(ref_frame.frame, 8))


def tensor_diff(a, b):
    """Return (max_abs_diff, mean_abs_diff) between two tensors."""
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def bytes_match(a, b):
    """Return True if two byte strings are identical."""
    return a == b


def first_differing_byte(a, b):
    """Return index of first byte that differs, or -1 if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential vs batched encoding frame-by-frame")
    parser.add_argument('--model_path_i', type=str, required=True)
    parser.add_argument('--model_path_p', type=str, required=True)
    parser.add_argument('--test_video',   type=str, required=True)
    parser.add_argument('--width',        type=int, default=640)
    parser.add_argument('--height',       type=int, default=360)
    parser.add_argument('--qp',           type=int, default=0)
    parser.add_argument('--num_frames',   type=int, default=100)
    parser.add_argument('--reset_interval', type=int, default=64,
                        help="Must match what test_video_batch.py uses")
    args = parser.parse_args()

    set_torch_env()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Video:  {args.test_video}  ({args.width}x{args.height})")
    print(f"QP:     {args.qp}   Frames: {args.num_frames}   Reset interval: {args.reset_interval}")
    print()

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    i_frame_net = DMCI()
    i_frame_net.load_state_dict(get_state_dict(args.model_path_i))
    i_frame_net = i_frame_net.to(device).eval()
    i_frame_net.half()
    i_frame_net.update()

    p_frame_net = DMC()
    p_frame_net.load_state_dict(get_state_dict(args.model_path_p))
    p_frame_net = p_frame_net.to(device).eval()
    p_frame_net.half()
    p_frame_net.update()

    w, h = args.width, args.height
    pad_r, pad_b = DMCI.get_padding_size(h, w, 16)
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    # -----------------------------------------------------------------------
    # Read all frames up front so I/O doesn't interfere with timing
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
    # Per-path state: one RefFrame per path to track temporal state
    # -----------------------------------------------------------------------
    ref_seq   = RefFrame()   # SEQUENTIAL path state
    ref_b1    = RefFrame()   # BATCH-1 path state
    ref_b2_0  = RefFrame()   # BATCH-2 path state, sequence slot 0
    ref_b2_1  = RefFrame()   # BATCH-2 path state, sequence slot 1 (duplicate)

    # Per-path last_qp for reset_interval handling
    last_qp_seq  = 0
    last_qp_b1   = 0
    last_qp_b2   = 0

    # Accumulators for summary
    first_mat_divergence   = None   # first frame where mat feature diverges
    first_bits_divergence  = None   # first frame where bitstream diverges
    first_feat_divergence  = None   # first frame where decoder output diverges
    first_b2_divergence    = None   # first frame where BATCH-2 diverges from SEQ

    print("=" * 90)
    print(f"{'Frame':>5}  {'Type':>4}  "
          f"{'MatFeat(B1-SEQ)':>18}  {'MatFeat(B2-SEQ)':>18}  "
          f"{'Bits match':>10}  "
          f"{'DecFeat(B1-SEQ)':>18}  {'DecFeat(B2-SEQ)':>18}")
    print("=" * 90)

    with torch.no_grad():
        for frame_idx in range(args.num_frames):
            x = frames[frame_idx]
            qp_i = args.qp
            fa_idx = index_map[frame_idx % 8]
            qp_p = p_frame_net.shift_qp(args.qp, fa_idx)
            is_i = (frame_idx == 0)

            # ------------------------------------------------------------------
            # I-FRAME: all three paths use the same i_frame_net.compress()
            # ------------------------------------------------------------------
            if is_i:
                encoded_i = i_frame_net.compress(x, qp_i)
                x_hat_i = encoded_i['x_hat']

                # All paths start with the same reconstructed I-frame
                ref_seq.frame  = x_hat_i;  ref_seq.feature  = None
                ref_b1.frame   = x_hat_i;  ref_b1.feature   = None
                ref_b2_0.frame = x_hat_i;  ref_b2_0.feature = None
                ref_b2_1.frame = x_hat_i;  ref_b2_1.feature = None

                last_qp_seq = last_qp_b1 = last_qp_b2 = qp_i
                print(f"{frame_idx:>5}  {'I':>4}  "
                      f"{'(I-frame, shared)':>18}  {'(I-frame, shared)':>18}  "
                      f"{'yes':>10}  "
                      f"{'(I-frame, shared)':>18}  {'(I-frame, shared)':>18}")
                continue

            # ------------------------------------------------------------------
            # RESET INTERVAL: mirrors test_video_batch.py lines 291-299
            # If frame_idx % reset_interval == 1, prepare_feature_adaptor_i
            # is called to regenerate the pixel reference from the feature.
            # We do this identically for all three paths.
            # ------------------------------------------------------------------
            if args.reset_interval > 0 and frame_idx % args.reset_interval == 1:
                for ref, lqp in [(ref_seq, last_qp_seq),
                                  (ref_b1,  last_qp_b1),
                                  (ref_b2_0, last_qp_b2),
                                  (ref_b2_1, last_qp_b2)]:
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(ref.feature, ref.frame)
                    p_frame_net.prepare_feature_adaptor_i(lqp)
                    ref.frame   = p_frame_net.dpb[0].frame
                    ref.feature = p_frame_net.dpb[0].feature

            # ------------------------------------------------------------------
            # MATERIALISE FEATURES for each path
            # This is the reference feature that will be fed into the encoder.
            # Comparing these first tells us if the *input* to the forward pass
            # is the same across all three paths.
            # ------------------------------------------------------------------
            mat_seq  = materialise_feature(p_frame_net, ref_seq)
            mat_b1   = materialise_feature(p_frame_net, ref_b1)
            mat_b2_0 = materialise_feature(p_frame_net, ref_b2_0)
            mat_b2_1 = materialise_feature(p_frame_net, ref_b2_1)

            mat_diff_b1_seq,  _ = tensor_diff(mat_b1,   mat_seq)
            mat_diff_b2_seq,  _ = tensor_diff(mat_b2_0, mat_seq)

            if first_mat_divergence is None and (mat_diff_b1_seq > 1e-4 or mat_diff_b2_seq > 1e-4):
                first_mat_divergence = frame_idx

            # ------------------------------------------------------------------
            # PATH 1: SEQUENTIAL — p_frame_net.compress() with internal DPB
            # ------------------------------------------------------------------
            p_frame_net.clear_dpb()
            p_frame_net.add_ref_frame(ref_seq.feature, ref_seq.frame)
            p_frame_net.set_curr_poc(frame_idx)
            enc_seq = p_frame_net.compress(x, qp_p)
            bits_seq     = enc_seq['bit_stream']
            feat_seq     = p_frame_net.dpb[0].feature.clone()
            # Update sequential ref state
            ref_seq.feature = feat_seq
            ref_seq.frame   = None
            last_qp_seq     = qp_p

            # ------------------------------------------------------------------
            # PATH 2: BATCH-1 — compress_batch() with B=1
            # Same single frame, same reference feature as sequential.
            # Any divergence here means compress_batch() has a bug vs compress().
            # ------------------------------------------------------------------
            batch_streams_b1, feat_out_b1, _ = p_frame_net.compress_batch(
                x, qp_p, mat_b1)
            bits_b1  = batch_streams_b1[0]
            feat_b1  = feat_out_b1[0:1].clone()
            # Update BATCH-1 ref state
            ref_b1.feature = feat_b1
            ref_b1.frame   = None
            last_qp_b1     = qp_p

            # ------------------------------------------------------------------
            # PATH 3: BATCH-2 — compress_batch() with B=2 (same frame twice)
            # Any divergence vs BATCH-1 means inter-sequence state leakage
            # when processing N>1 sequences in a single batch.
            # ------------------------------------------------------------------
            x_batch    = torch.cat([x, x], dim=0)
            mat_b2     = torch.cat([mat_b2_0, mat_b2_1], dim=0)
            batch_streams_b2, feat_out_b2, _ = p_frame_net.compress_batch(
                x_batch, qp_p, mat_b2)
            bits_b2_0  = batch_streams_b2[0]
            bits_b2_1  = batch_streams_b2[1]
            feat_b2_0  = feat_out_b2[0:1].clone()
            feat_b2_1  = feat_out_b2[1:2].clone()
            # Update BATCH-2 ref state (use slot 0 as representative)
            ref_b2_0.feature = feat_b2_0
            ref_b2_0.frame   = None
            ref_b2_1.feature = feat_b2_1
            ref_b2_1.frame   = None
            last_qp_b2       = qp_p

            # ------------------------------------------------------------------
            # COMPARE RESULTS
            # ------------------------------------------------------------------

            # Bitstream comparison
            # Checks if the entropy coder produces identical bytes.
            # Note: BATCH-2 slot 0 and slot 1 should also match each other since
            # they received identical inputs.
            bits_b1_match_seq  = bytes_match(bits_b1,   bits_seq)
            bits_b2_match_seq  = bytes_match(bits_b2_0, bits_seq)
            bits_b2_internal   = bytes_match(bits_b2_0, bits_b2_1)  # sanity: both slots identical?

            if first_bits_divergence is None and not (bits_b1_match_seq and bits_b2_match_seq):
                first_bits_divergence = frame_idx

            # Decoder output feature comparison
            # These features are stored in ref_frames and used as reference for
            # the next frame. Any difference here will compound over time.
            feat_diff_b1_seq,  feat_mean_b1  = tensor_diff(feat_b1,   feat_seq)
            feat_diff_b2_seq,  feat_mean_b2  = tensor_diff(feat_b2_0, feat_seq)
            feat_diff_b2_int,  _             = tensor_diff(feat_b2_0, feat_b2_1)  # sanity

            if first_feat_divergence is None and feat_diff_b1_seq > 1e-4:
                first_feat_divergence = frame_idx
            if first_b2_divergence is None and feat_diff_b2_seq > 1e-4:
                first_b2_divergence = frame_idx

            # ------------------------------------------------------------------
            # Print per-frame summary row
            # Format: frame | type | mat_diff_b1 | mat_diff_b2 | bits_match | feat_diff_b1 | feat_diff_b2
            # ------------------------------------------------------------------
            bits_col = ("yes" if (bits_b1_match_seq and bits_b2_match_seq and bits_b2_internal)
                        else f"B1:{int(bits_b1_match_seq)} B2:{int(bits_b2_match_seq)} int:{int(bits_b2_internal)}")

            # Flag frames with significant divergence with a marker
            flag = ""
            if mat_diff_b1_seq > 1e-4 or feat_diff_b1_seq > 1e-4:
                flag += " !! B1"
            if mat_diff_b2_seq > 1e-4 or feat_diff_b2_seq > 1e-4:
                flag += " !! B2"
            if feat_diff_b2_int > 1e-4:
                flag += " !! B2-int"

            print(f"{frame_idx:>5}  {'P':>4}  "
                  f"{mat_diff_b1_seq:>18.6e}  {mat_diff_b2_seq:>18.6e}  "
                  f"{bits_col:>10}  "
                  f"{feat_diff_b1_seq:>18.6e}  {feat_diff_b2_seq:>18.6e}"
                  f"{flag}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"First frame where materialised feature diverges (B1 or B2 vs SEQ): "
          f"{first_mat_divergence if first_mat_divergence is not None else 'never'}")
    print(f"First frame where bitstream diverges (B1 or B2 vs SEQ):            "
          f"{first_bits_divergence if first_bits_divergence is not None else 'never'}")
    print(f"First frame where decoder output feature diverges (B1 vs SEQ):     "
          f"{first_feat_divergence if first_feat_divergence is not None else 'never'}")
    print(f"First frame where decoder output feature diverges (B2 vs SEQ):     "
          f"{first_b2_divergence if first_b2_divergence is not None else 'never'}")
    print()

    # Interpret the findings
    if first_mat_divergence is None and first_feat_divergence is not None:
        print("DIAGNOSIS: Materialised features are identical (inputs are correct),")
        print("           but compress_batch(B=1) produces different decoder output features.")
        print("           => BUG IS INSIDE compress_batch() FORWARD PASS vs compress().")

    elif first_mat_divergence is not None:
        print("DIAGNOSIS: Materialised features diverge — the reference state stored in")
        print("           ref_frames is being corrupted between frames.")
        print("           => BUG IS IN HOW ref_frames ARE UPDATED AFTER EACH FRAME.")

    elif first_bits_divergence is not None and first_feat_divergence is None:
        print("DIAGNOSIS: Decoder output features match but bitstreams diverge.")
        print("           => BUG IS IN THE ENTROPY CODER STATE between calls.")

    elif first_b2_divergence is not None and first_feat_divergence is None:
        print("DIAGNOSIS: BATCH-1 matches SEQ perfectly, but BATCH-2 diverges.")
        print("           => BUG IS INTER-SEQUENCE STATE LEAKAGE in B=2 path.")

    elif first_feat_divergence is None and first_b2_divergence is None:
        print("DIAGNOSIS: All three paths match perfectly.")
        print("           => The encoding logic is correct. Bug may be in test harness.")
    else:
        print("DIAGNOSIS: Multiple issues detected. Review per-frame output above.")


if __name__ == "__main__":
    main()

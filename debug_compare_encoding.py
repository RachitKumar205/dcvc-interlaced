#!/usr/bin/env python3
"""
debug_compare_encoding.py

Encodes the same frame(s) using both sequential and batched paths,
then saves intermediate tensors for comparison.

Usage:
    python debug_compare_encoding.py \
        --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
        --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
        --test_video test_data/UVG/ShakeNDry_640x360_120fps_420_8bit_YUV.yuv \
        --qp 0
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.models.common_model import RefFrame
from src.utils.common import get_state_dict, set_torch_env
from src.utils.yuv import YUV420Reader


def ycbcr420_to_444_np(y, uv):
    h, w = y.shape
    u = uv[0, :, :]
    v = uv[1, :, :]
    yuv = np.stack([y, u, v], axis=0)
    return yuv


def replicate_pad(x, pad_b, pad_r):
    if pad_b > 0 or pad_r > 0:
        return F.pad(x, (0, pad_r, 0, pad_b), mode='replicate')
    return x


def materialise_feature(p_frame_net, ref_frame):
    if ref_frame.feature is not None:
        return p_frame_net.feature_adaptor_p(ref_frame.feature)
    return p_frame_net.feature_adaptor_i(F.pixel_unshuffle(ref_frame.frame, 8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path_i', type=str, required=True)
    parser.add_argument('--model_path_p', type=str, required=True)
    parser.add_argument('--test_video', type=str, required=True)
    parser.add_argument('--qp', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    args = parser.parse_args()

    set_torch_env()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load models
    i_frame_net = DMCI()
    i_frame_net.load_state_dict(get_state_dict(args.model_path_i))
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    i_frame_net.half()

    p_frame_net = DMC()
    p_frame_net.load_state_dict(get_state_dict(args.model_path_p))
    p_frame_net = p_frame_net.to(device)
    p_frame_net.eval()
    p_frame_net.half()

    w, h = args.width, args.height
    padding_r, padding_b = DMCI.get_padding_size(h, w, 16)

    # Read first 2 frames
    reader = YUV420Reader(args.test_video, w, h)
    y0, uv0 = reader.read_one_frame()
    y1, uv1 = reader.read_one_frame()
    reader.close()

    yuv0 = ycbcr420_to_444_np(y0, uv0)
    yuv1 = ycbcr420_to_444_np(y1, uv1)

    x0 = torch.from_numpy(yuv0).to(device=device, dtype=torch.float16).unsqueeze(0) / 255.0
    x1 = torch.from_numpy(yuv1).to(device=device, dtype=torch.float16).unsqueeze(0) / 255.0

    x0_padded = replicate_pad(x0, padding_b, padding_r)
    x1_padded = replicate_pad(x1, padding_b, padding_r)

    qp = args.qp

    print("=" * 70)
    print("SEQUENTIAL ENCODING")
    print("=" * 70)

    # --- Frame 0: I-frame (sequential) ---
    with torch.no_grad():
        i_encoded = i_frame_net.compress(x0_padded, qp)
    i_x_hat = i_encoded['x_hat']
    print(f"Frame 0 (I): x_hat shape = {i_x_hat.shape}")

    # Store ref frame after I-frame
    ref_seq = RefFrame()
    ref_seq.feature = None
    ref_seq.frame = i_x_hat

    # --- Frame 1: P-frame (sequential) ---
    with torch.no_grad():
        p_frame_net.clear_dpb()
        p_frame_net.add_ref_frame(ref_seq.feature, ref_seq.frame)
        p_frame_net.set_curr_poc(0)

        # Materialise feature for sequential
        seq_mat_feature = materialise_feature(p_frame_net, ref_seq)
        print(f"Sequential materialised feature shape: {seq_mat_feature.shape}")

        # Use regular compress
        p_frame_net.clear_dpb()
        p_frame_net.add_ref_frame(ref_seq.feature, ref_seq.frame)
        seq_encoded = p_frame_net.compress(x1_padded, qp)
        seq_feature = p_frame_net.dpb[0].feature.clone()
        print(f"Sequential P-frame output feature shape: {seq_feature.shape}")

    # Save sequential results
    torch.save({
        'i_x_hat': i_x_hat.cpu().float(),
        'seq_mat_feature': seq_mat_feature.cpu().float(),
        'seq_feature': seq_feature.cpu().float(),
    }, 'debug_sequential.pt')
    print("\nSaved sequential results to debug_sequential.pt")

    print("\n" + "=" * 70)
    print("BATCHED ENCODING (B=2, same frame twice)")
    print("=" * 70)

    # --- Frame 0: I-frame (same as sequential) ---
    # Already done above, reuse i_x_hat

    ref_batch_0 = RefFrame()
    ref_batch_0.feature = None
    ref_batch_0.frame = i_x_hat

    ref_batch_1 = RefFrame()
    ref_batch_1.feature = None
    ref_batch_1.frame = i_x_hat  # Same I-frame for both

    # --- Frame 1: P-frame (batched B=2) ---
    with torch.no_grad():
        # Stack inputs
        x_batch = torch.cat([x1_padded, x1_padded], dim=0)

        # Materialise features
        mat_f0 = materialise_feature(p_frame_net, ref_batch_0)
        mat_f1 = materialise_feature(p_frame_net, ref_batch_1)
        mat_features = torch.cat([mat_f0, mat_f1], dim=0)
        print(f"Batched materialised features shape: {mat_features.shape}")

        # Compare materialised features
        mat_diff = (seq_mat_feature - mat_f0).abs().max().item()
        print(f"Materialised feature diff (seq vs batch): {mat_diff:.6e}")

        p_frame_net.set_curr_poc(0)
        batch_streams, features_out, timing = p_frame_net.compress_batch(
            x_batch, qp, mat_features)

        batch_feature_0 = features_out[0:1].clone()
        batch_feature_1 = features_out[1:2].clone()
        print(f"Batched P-frame output features shape: {features_out.shape}")

    # Save batched results
    torch.save({
        'batch_mat_features': mat_features.cpu().float(),
        'batch_feature_0': batch_feature_0.cpu().float(),
        'batch_feature_1': batch_feature_1.cpu().float(),
        'timing': timing,
    }, 'debug_batched.pt')
    print("\nSaved batched results to debug_batched.pt")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Load and compare
    seq_data = torch.load('debug_sequential.pt', weights_only=True)
    batch_data = torch.load('debug_batched.pt', weights_only=True)

    seq_feat = seq_data['seq_feature']
    batch_feat_0 = batch_data['batch_feature_0']
    batch_feat_1 = batch_data['batch_feature_1']

    diff_0 = (seq_feat - batch_feat_0).abs().max().item()
    diff_1 = (seq_feat - batch_feat_1).abs().max().item()
    mean_diff_0 = (seq_feat - batch_feat_0).abs().mean().item()
    mean_diff_1 = (seq_feat - batch_feat_1).abs().mean().item()

    print(f"Feature diff (seq vs batch[0]): max={diff_0:.6e}, mean={mean_diff_0:.6e}")
    print(f"Feature diff (seq vs batch[1]): max={diff_1:.6e}, mean={mean_diff_1:.6e}")
    print(f"Feature diff (batch[0] vs batch[1]): max={(batch_feat_0 - batch_feat_1).abs().max().item():.6e}")

    print(f"\nTiming: GPU={timing['gpu_forward']*1000:.2f}ms, CPU={timing['cpu_entropy']*1000:.2f}ms")

    if diff_0 > 1e-5:
        print("\n*** MISMATCH DETECTED: Batched features differ from sequential! ***")
    else:
        print("\n*** MATCH: Batched features are identical to sequential. ***")


if __name__ == "__main__":
    main()

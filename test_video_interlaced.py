# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DCVC-BC: Micro-Interlaced encoding extension.
#
# The standard DCVC-RT pipeline is strictly sequential: encoding frame t
# requires the decoded feature from frame t-1. This file implements
# Micro-Interlaced coding, which breaks that dependency by splitting the
# video into two independent sub-sequences:
#
#   Stream A (even frames): F0 -> F2 -> F4 -> ...
#   Stream B (odd  frames): F1 -> F3 -> F5 -> ...
#
# Each stream is an independent autoregressive chain. Within each pair
# (F_{2i}, F_{2i+1}), the two compress() calls have no dependency on each
# other and are dispatched to separate CUDA streams so they can overlap on
# the GPU.
#
# The only quality cost is the "reference gap": frame t now references t-2
# instead of t-1, weakening temporal prediction slightly.
#
# Usage is identical to test_video.py with one additional flag:
#   --interlaced 1

import argparse
import concurrent.futures
import io
import json
import multiprocessing
import os
import time
import threading

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.layers.cuda_inference import replicate_pad
from src.models.video_model import DMC, RefFrame
from src.models.image_model import DMCI
from src.utils.common import str2bool, create_folder, generate_log_json, get_state_dict, \
    dump_json, set_torch_env
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.video_writer import PNGWriter, YUV420Writer
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np


def parse_args():
    parser = argparse.ArgumentParser(description="DCVC-BC Micro-Interlaced testing script")

    parser.add_argument('--force_zero_thres', type=float, default=None, required=False)
    parser.add_argument('--model_path_i', type=str)
    parser.add_argument('--model_path_p',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--qp_i', type=int, nargs="+")
    parser.add_argument('--qp_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, default=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    if args['src_type'] == 'png':
        src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])
    elif args['src_type'] == 'yuv420':
        src_reader = YUV420Reader(args['src_path'], args['src_width'], args['src_height'])
    return src_reader


def get_src_frame(args, src_reader, device):
    if args['src_type'] == 'yuv420':
        y, uv = src_reader.read_one_frame()
        yuv = ycbcr420_to_444_np(y, uv)
        x = np_image_to_tensor(yuv, device)
        y = y[0, :, :]
        u = uv[0, :, :]
        v = uv[1, :, :]
        rgb = None
    else:
        assert args['src_type'] == 'png'
        rgb = src_reader.read_one_frame()
        x = np_image_to_tensor(rgb, device)
        x = rgb2ycbcr(x)
        y, u, v = None, None, None

    x = x.to(torch.float16)
    return x, y, u, v, rgb


def get_distortion(args, x_hat, y, u, v, rgb):
    if args['src_type'] == 'yuv420':
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]
        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        if args['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec)
            ssim_u = calc_msssim(u, u_rec)
            ssim_v = calc_msssim(v, v_rec)
        else:
            ssim_y, ssim_u, ssim_v = 0., 0., 0.
        ssim = (6 * ssim_y + ssim_u + ssim_v) / 8
        curr_psnr = [psnr, psnr_y, psnr_u, psnr_v]
        curr_ssim = [ssim, ssim_y, ssim_u, ssim_v]
    else:
        assert args['src_type'] == 'png'
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        psnr = calc_psnr(rgb, rgb_rec)
        if args['calc_ssim']:
            msssim = calc_msssim_rgb(rgb, rgb_rec)
        else:
            msssim = 0.
        curr_psnr = [psnr]
        curr_ssim = [msssim]
    return curr_psnr, curr_ssim


# ---------------------------------------------------------------------------
# DPB state helpers
# ---------------------------------------------------------------------------

def save_dpb_state(p_frame_net):
    """Snapshot the DPB and POC counter so we can restore them later."""
    return {
        'dpb': list(p_frame_net.dpb),
        'curr_poc': p_frame_net.curr_poc,
    }


def restore_dpb_state(p_frame_net, state):
    """Restore a previously saved DPB snapshot."""
    p_frame_net.dpb = list(state['dpb'])
    p_frame_net.curr_poc = state['curr_poc']


def materialise_feature(p_frame_net, ref_frame):
    """
    Return the [1, g_ch_d, H, W] reference feature for a stream's RefFrame,
    mirroring what DMC.apply_feature_adaptor() does internally but without
    touching p_frame_net.dpb.

    After a P-frame: ref_frame.feature is set — run through feature_adaptor_p.
    After an I-frame: ref_frame.feature is None, ref_frame.frame holds x_hat
                      — run through feature_adaptor_i after pixel_unshuffle.
    """
    if ref_frame.feature is not None:
        return p_frame_net.feature_adaptor_p(ref_frame.feature)
    return p_frame_net.feature_adaptor_i(F.pixel_unshuffle(ref_frame.frame, 8))


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _write_pair(bit_stream_a, bit_stream_b, is_i_a, is_i_b,
                qp_a, qp_b, sps_a, sps_b, has_odd,
                even_orig_idx, odd_orig_idx,
                bits, frame_types, sps_helper, output_buff):
    sps_id_a, sps_new_a = sps_helper.get_sps_id(sps_a)
    sps_a['sps_id'] = sps_id_a
    sps_bytes_a = write_sps(output_buff, sps_a) if sps_new_a else 0
    stream_bytes_a = write_ip(output_buff, is_i_a, sps_id_a, qp_a, bit_stream_a)
    bits[even_orig_idx] = (stream_bytes_a + sps_bytes_a) * 8
    frame_types[even_orig_idx] = 0 if is_i_a else 1
    if has_odd:
        sps_id_b, sps_new_b = sps_helper.get_sps_id(sps_b)
        sps_b['sps_id'] = sps_id_b
        sps_bytes_b = write_sps(output_buff, sps_b) if sps_new_b else 0
        stream_bytes_b = write_ip(output_buff, is_i_b, sps_id_b, qp_b, bit_stream_b)
        bits[odd_orig_idx] = (stream_bytes_b + sps_bytes_b) * 8
        frame_types[odd_orig_idx] = 0 if is_i_b else 1


def _record_time(pair_time, has_odd, even_orig_idx, odd_orig_idx, encoding_times, bits, verbose):
    per_frame_time = pair_time / (2 if has_odd else 1)
    encoding_times[even_orig_idx] = per_frame_time
    if has_odd:
        encoding_times[odd_orig_idx] = per_frame_time
    if verbose >= 2:
        n = 2 if has_odd else 1
        print(f"  pair [{even_orig_idx},{odd_orig_idx if has_odd else '-'}] "
              f"{pair_time*1000:.1f} ms total, {per_frame_time*1000:.1f} ms/frame")


def _collect_and_write(p_frame_net, pending_p, pending_meta,
                       bits, frame_types, encoding_times,
                       sps_helper, output_buff, verbose):
    """Block on entropy workers, sync GPU, write bitstreams, record timing."""
    bit_stream_a, bit_stream_b, feat_a, feat_b = \
        p_frame_net.compress_batch_collect(pending_p)
    m = pending_meta
    _write_pair(bit_stream_a, bit_stream_b, m['is_i_a'], m['is_i_b'],
                m['qp_a'], m['qp_b'], m['sps_a'], m['sps_b'], m['has_odd'],
                m['even_orig_idx'], m['odd_orig_idx'],
                bits, frame_types, sps_helper, output_buff)
    pair_time = time.time() - m['pair_start']
    _record_time(pair_time, m['has_odd'], m['even_orig_idx'], m['odd_orig_idx'],
                 encoding_times, bits, verbose)
    return feat_a, feat_b


# ---------------------------------------------------------------------------
# Interlaced encode
# ---------------------------------------------------------------------------

def encode_interlaced(p_frame_net, i_frame_net, args, frames, pic_height, pic_width,
                      padding_r, padding_b, use_two_entropy_coders, sps_helper, output_buff):
    """
    Encode all frames using micro-interlaced coding.

    The video is split into Stream A (even indices) and Stream B (odd indices).
    Each stream is an independent autoregressive chain.

    I-frame pairs (pair_idx == 0) are encoded sequentially — they are not
    batchable because i_frame_net.compress() has a stateful entropy coder.

    All P-frame pairs use p_frame_net.compress_batch(): a single B=2 GPU
    forward pass followed by parallel entropy coding on two threads.

    Returns:
        frame_types  - list of 0 (I) or 1 (P) per original frame index
        bits         - list of bit counts per original frame index
        encoding_times - list of wall-clock encode time per original frame index
    """
    frame_num = len(frames)
    device = next(i_frame_net.parameters()).device
    reset_interval = args['reset_interval']
    verbose = args['verbose']

    frame_types = [0] * frame_num
    bits = [0] * frame_num
    encoding_times = [0.0] * frame_num

    # Per-stream reference state.  After an I-frame: ref.feature is None,
    # ref.frame holds x_hat.  After a P-frame: ref.feature holds the decoder
    # output feature, ref.frame is None.
    ref_a = RefFrame()
    ref_b = RefFrame()

    # last QP per stream (needed for prepare_feature_adaptor_i)
    last_qp_a = 0
    last_qp_b = 0

    # Position of the current frame within each sub-sequence
    sub_idx_a = 0
    sub_idx_b = 0

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    def make_sps(use_ada_i):
        return {
            'sps_id': -1,
            'height': pic_height,
            'width': pic_width,
            'ec_part': 1 if use_two_entropy_coders else 0,
            'use_ada_i': use_ada_i,
        }

    pair_count = (frame_num + 1) // 2

    # Pipeline state: after compress_batch_async() returns, we immediately use
    # the (still-in-flight) GPU feature tensor as the next pair's reference.
    # PyTorch enforces the GPU-side ordering automatically because all GPU ops
    # go to the default stream — the next pair's GPU forward queues behind the
    # current pair's decoder. Meanwhile collect() blocks on the CPU entropy
    # workers, which overlap with that next-pair GPU forward.
    #
    # pending_p: the async pending dict for the previous P-frame pair, to be
    #            collected at the start of the next iteration (or after the loop).
    # pending_meta: sps/qp/idx metadata needed to write the bitstream after collect.
    pending_p = None
    pending_meta = None

    for pair_idx in range(pair_count):
        even_orig_idx = pair_idx * 2
        odd_orig_idx  = pair_idx * 2 + 1

        x_even, _, _, _, _ = frames[even_orig_idx]
        has_odd = (odd_orig_idx < frame_num)
        if has_odd:
            x_odd, _, _, _, _ = frames[odd_orig_idx]

        # Start timing. For pipelined P-pairs, the sync is replaced by the
        # GPU-side event ordering — but we still sync here to get a meaningful
        # wall-clock number for I-frames and the last odd frame.
        if torch.cuda.is_available() and (pending_p is None):
            torch.cuda.synchronize(device=device)
        pair_start = time.time()

        x_even_padded = replicate_pad(x_even, padding_b, padding_r)
        if has_odd:
            x_odd_padded = replicate_pad(x_odd, padding_b, padding_r)

        is_i_a = (sub_idx_a == 0)
        is_i_b = (sub_idx_b == 0)
        bit_stream_b = b''
        qp_b = 0
        sps_b = make_sps(0)

        if is_i_a:
            # ---------------------------------------------------------------
            # I-frame pair — encode sequentially (not batchable).
            # If there's a pending P-pair from before, collect it first.
            # ---------------------------------------------------------------
            if pending_p is not None:
                _collect_and_write(p_frame_net, pending_p, pending_meta,
                                   bits, frame_types, encoding_times, sps_helper,
                                   output_buff, verbose)
                pending_p = None
                pending_meta = None

            qp_a = args['qp_i']
            sps_a = make_sps(0)
            enc_a = i_frame_net.compress(x_even_padded, qp_a)
            ref_a.frame = enc_a['x_hat']
            ref_a.feature = None
            bit_stream_a = enc_a['bit_stream']

            if has_odd:
                qp_b = args['qp_i']
                sps_b = make_sps(0)
                enc_b = i_frame_net.compress(x_odd_padded, qp_b)
                ref_b.frame = enc_b['x_hat']
                ref_b.feature = None
                bit_stream_b = enc_b['bit_stream']

            _write_pair(bit_stream_a, bit_stream_b, is_i_a, is_i_b,
                        qp_a, qp_b, sps_a, sps_b, has_odd,
                        even_orig_idx, odd_orig_idx,
                        bits, frame_types, sps_helper, output_buff)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
            pair_end = time.time()
            _record_time(pair_end - pair_start, has_odd,
                         even_orig_idx, odd_orig_idx, encoding_times, bits, verbose)

        elif has_odd:
            # ---------------------------------------------------------------
            # P-frame pair — pipelined: async forward now, collect previous.
            # ---------------------------------------------------------------
            fa_idx = index_map[sub_idx_a % 8]
            use_ada_i = 0

            if reset_interval > 0 and sub_idx_a % reset_interval == 1:
                # Must collect any pending pair before mutating DPB.
                if pending_p is not None:
                    _collect_and_write(p_frame_net, pending_p, pending_meta,
                                       bits, frame_types, encoding_times, sps_helper,
                                       output_buff, verbose)
                    pending_p = None
                    pending_meta = None
                use_ada_i = 1
                for ref, last_qp in ((ref_a, last_qp_a), (ref_b, last_qp_b)):
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(ref.feature, ref.frame)
                    p_frame_net.prepare_feature_adaptor_i(last_qp)
                    ref.frame = p_frame_net.dpb[0].frame
                    ref.feature = p_frame_net.dpb[0].feature

            curr_qp = p_frame_net.shift_qp(args['qp_p'], fa_idx)
            sps_a = sps_b = make_sps(use_ada_i)
            qp_a = qp_b = curr_qp

            x_batch = torch.cat((x_even_padded, x_odd_padded), dim=0)

            # Launch this pair's GPU forward. Pass the previous decoder event
            # so PyTorch inserts a GPU-side wait before our GPU ops — no CPU sync.
            prev_event = pending_p['decoder_event'] if pending_p is not None else None
            new_pending = p_frame_net.compress_batch_async(
                x_batch, curr_qp, ref_a, ref_b, prev_decoder_event=prev_event)

            # Immediately use the (in-flight) GPU feature tensor as refs for the
            # next pair. The GPU-side ordering ensures correctness.
            ref_a.feature = new_pending['feature'][0:1]
            ref_a.frame = None
            ref_b.feature = new_pending['feature'][1:2]
            ref_b.frame = None
            last_qp_a = last_qp_b = curr_qp

            # Now collect the PREVIOUS pair (its entropy workers have been running
            # while we launched this pair's GPU forward above).
            if pending_p is not None:
                _collect_and_write(p_frame_net, pending_p, pending_meta,
                                   bits, frame_types, encoding_times, sps_helper,
                                   output_buff, verbose)

            pending_p = new_pending
            pending_meta = {
                'sps_a': sps_a, 'sps_b': sps_b,
                'qp_a': qp_a,   'qp_b': qp_b,
                'is_i_a': is_i_a, 'is_i_b': is_i_b,
                'has_odd': has_odd,
                'even_orig_idx': even_orig_idx,
                'odd_orig_idx': odd_orig_idx,
                'pair_start': pair_start,
            }

        else:
            # ---------------------------------------------------------------
            # Last frame of an odd-length video — single P-frame, no batch.
            # ---------------------------------------------------------------
            if pending_p is not None:
                _collect_and_write(p_frame_net, pending_p, pending_meta,
                                   bits, frame_types, encoding_times, sps_helper,
                                   output_buff, verbose)
                pending_p = None
                pending_meta = None

            fa_idx = index_map[sub_idx_a % 8]
            use_ada_i = 0

            if reset_interval > 0 and sub_idx_a % reset_interval == 1:
                use_ada_i = 1
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(ref_a.feature, ref_a.frame)
                p_frame_net.prepare_feature_adaptor_i(last_qp_a)
                ref_a.frame = p_frame_net.dpb[0].frame
                ref_a.feature = p_frame_net.dpb[0].feature

            curr_qp = p_frame_net.shift_qp(args['qp_p'], fa_idx)
            sps_a = make_sps(use_ada_i)
            qp_a = curr_qp

            p_frame_net.clear_dpb()
            p_frame_net.add_ref_frame(ref_a.feature, ref_a.frame)
            enc_a = p_frame_net.compress(x_even_padded, curr_qp)
            ref_a.feature = p_frame_net.dpb[0].feature
            ref_a.frame = None
            last_qp_a = curr_qp
            bit_stream_a = enc_a['bit_stream']

            _write_pair(bit_stream_a, b'', is_i_a, False,
                        qp_a, 0, sps_a, make_sps(0), False,
                        even_orig_idx, odd_orig_idx,
                        bits, frame_types, sps_helper, output_buff)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
            pair_end = time.time()
            _record_time(pair_end - pair_start, False,
                         even_orig_idx, odd_orig_idx, encoding_times, bits, verbose)

        sub_idx_a += 1
        if has_odd:
            sub_idx_b += 1

    # Collect the last pending P-pair.
    if pending_p is not None:
        _collect_and_write(p_frame_net, pending_p, pending_meta,
                           bits, frame_types, encoding_times, sps_helper,
                           output_buff, verbose)

    return frame_types, bits, encoding_times


# ---------------------------------------------------------------------------
# Interlaced decode
# ---------------------------------------------------------------------------

def decode_interlaced(p_frame_net, i_frame_net, args, input_buff, frame_num,
                      pic_height, pic_width, sps_helper):
    """
    Decode a bitstream produced by encode_interlaced.

    Frames arrive interleaved (A0, B0, A1, B1, ...). We route even-indexed
    frames to DPB state A and odd-indexed frames to DPB state B.

    Returns:
        recon_frames   - list of x_hat tensors in original frame order
        decoding_times - list of wall-clock decode time per frame
    """
    device = next(i_frame_net.parameters()).device
    verbose = args['verbose']

    recon_frames = [None] * frame_num
    decoding_times = [None] * frame_num

    state_a = None
    state_b = None

    p_frame_net.clear_dpb()
    state_a = save_dpb_state(p_frame_net)
    state_b = save_dpb_state(p_frame_net)

    for frame_idx in range(frame_num):
        is_even = (frame_idx % 2 == 0)
        state = state_a if is_even else state_b

        torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
        frame_start = time.time()

        restore_dpb_state(p_frame_net, state)

        header = read_header(input_buff)
        while header['nal_type'] == NalType.NAL_SPS:
            sps = read_sps_remaining(input_buff, header['sps_id'])
            sps_helper.add_sps_by_id(sps)
            if verbose >= 2:
                print("new sps", sps)
            header = read_header(input_buff)

        sps_id = header['sps_id']
        sps = sps_helper.get_sps_by_id(sps_id)
        qp, bit_stream = read_ip_remaining(input_buff)

        if header['nal_type'] == NalType.NAL_I:
            decoded = i_frame_net.decompress(bit_stream, sps, qp)
            p_frame_net.clear_dpb()
            p_frame_net.add_ref_frame(None, decoded['x_hat'])
        elif header['nal_type'] == NalType.NAL_P:
            if sps['use_ada_i']:
                p_frame_net.reset_ref_feature()
            decoded = p_frame_net.decompress(bit_stream, sps, qp)

        # Save updated DPB back to the correct stream state
        if is_even:
            state_a = save_dpb_state(p_frame_net)
        else:
            state_b = save_dpb_state(p_frame_net)

        torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
        frame_end = time.time()

        recon_frames[frame_idx] = decoded['x_hat']
        decoding_times[frame_idx] = frame_end - frame_start

        if verbose >= 2:
            stream_length = 0 if bit_stream is None else len(bit_stream) * 8
            print(f"frame {frame_idx} decoded, {(frame_end-frame_start)*1000:.3f} ms, "
                  f"bits: {stream_length}")

    return recon_frames, decoding_times


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def run_one_point_with_stream(p_frame_net, i_frame_net, args):
    if args['check_existing'] and os.path.exists(args['curr_json_path']) and \
            os.path.exists(args['curr_bin_path']):
        with open(args['curr_json_path']) as f:
            log_result = json.load(f)
            if log_result['i_frame_num'] + log_result['p_frame_num'] == args['frame_num']:
                return log_result
            print(f"incorrect log for {args['curr_json_path']}, try to rerun.")

    frame_num = args['frame_num']
    save_decoded_frame = args['save_decoded_frame']
    verbose = args['verbose']
    verbose_json = args['verbose_json']
    device = next(i_frame_net.parameters()).device

    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

    use_two_entropy_coders = pic_height * pic_width > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    start_time = time.time()

    # --- Read all frames upfront ---
    # We need random access to frames for interlaced pairing.
    src_reader = get_src_reader(args)
    frames = []
    for _ in range(frame_num):
        frame_data = get_src_frame(args, src_reader, device)
        frames.append(frame_data)
    src_reader.close()

    # --- Encode ---
    output_buff = io.BytesIO()
    sps_helper = SPSHelper()
    p_frame_net.set_curr_poc(0)

    with torch.no_grad():
        frame_types, bits, encoding_times = encode_interlaced(
            p_frame_net, i_frame_net, args, frames,
            pic_height, pic_width, padding_r, padding_b,
            use_two_entropy_coders, sps_helper, output_buff
        )

    # Write bitstream to disk
    with open(args['curr_bin_path'], "wb") as output_file:
        bytes_buffer = output_buff.getbuffer()
        output_file.write(bytes_buffer)
        total_bytes = bytes_buffer.nbytes
        bytes_buffer.release()
    total_kbps = int(total_bytes * 8 / (frame_num / 30) / 1000)
    output_buff.close()

    # --- Decode ---
    sps_helper = SPSHelper()
    with open(args['curr_bin_path'], "rb") as input_file:
        input_buff = io.BytesIO(input_file.read())

    if save_decoded_frame:
        if args['src_type'] == 'png':
            recon_writer = PNGWriter(args['bin_folder'], args['src_width'], args['src_height'])
        elif args['src_type'] == 'yuv420':
            output_yuv_path = args['curr_rec_path'].replace('.yuv', f'_{total_kbps}kbps.yuv')
            recon_writer = YUV420Writer(output_yuv_path, args['src_width'], args['src_height'])

    p_frame_net.set_curr_poc(0)
    with torch.no_grad():
        recon_frames, decoding_times = decode_interlaced(
            p_frame_net, i_frame_net, args, input_buff, frame_num,
            pic_height, pic_width, sps_helper
        )
    input_buff.close()

    # --- Compute distortion ---
    psnrs = []
    msssims = []
    for frame_idx in range(frame_num):
        _, y, u, v, rgb = frames[frame_idx]
        x_hat = recon_frames[frame_idx]
        x_hat_cropped = x_hat[:, :, :pic_height, :pic_width]
        curr_psnr, curr_ssim = get_distortion(args, x_hat_cropped, y, u, v, rgb)
        psnrs.append(curr_psnr)
        msssims.append(curr_ssim)

        if save_decoded_frame:
            if args['src_type'] == 'yuv420':
                y_rec, uv_rec = yuv_444_to_420(x_hat_cropped)
                y_rec = torch.clamp(y_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                y_rec = y_rec.squeeze(0).cpu().numpy()
                uv_rec = torch.clamp(uv_rec * 255, 0, 255).to(dtype=torch.uint8)
                uv_rec = uv_rec.squeeze(0).cpu().numpy()
                recon_writer.write_one_frame(y_rec, uv_rec)
            else:
                rgb_rec = ycbcr2rgb(x_hat_cropped)
                rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                rgb_rec = rgb_rec.squeeze(0).cpu().numpy()
                recon_writer.write_one_frame(rgb_rec)

    if save_decoded_frame:
        recon_writer.close()

    test_time = time.time() - start_time

    # --- Timing report ---
    time_bypass = 10
    avg_encoding_time = None
    avg_decoding_time = None
    if verbose >= 1 and frame_num > time_bypass:
        enc_times = encoding_times[time_bypass:]
        dec_times = decoding_times[time_bypass:]
        avg_encoding_time = sum(enc_times) / len(enc_times)
        avg_decoding_time = sum(dec_times) / len(dec_times)
        print(f"[interlaced] encoding/decoding {frame_num} frames, "
              f"avg encoding time {avg_encoding_time*1000:.3f} ms/frame, "
              f"avg decoding time {avg_decoding_time*1000:.3f} ms/frame")

    log_result = generate_log_json(frame_num, pic_height * pic_width, test_time,
                                   frame_types, bits, psnrs, msssims, verbose=verbose_json,
                                   avg_encoding_time=avg_encoding_time,
                                   avg_decoding_time=avg_decoding_time)
    with open(args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    return log_result


# ---------------------------------------------------------------------------
# Worker / process pool (identical to test_video.py)
# ---------------------------------------------------------------------------

i_frame_net = None
p_frame_net = None


def worker(args):
    global i_frame_net
    global p_frame_net

    sub_dir_name = args['seq']
    bin_folder = os.path.join(args['stream_path'], args['ds_name'])
    assert args['write_stream'], ""
    create_folder(bin_folder, True)

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['curr_bin_path'] = os.path.join(bin_folder, f"{args['seq']}_q{args['qp_i']}.bin")
    args['curr_rec_path'] = args['curr_bin_path'].replace('.bin', '.yuv')
    args['curr_json_path'] = args['curr_bin_path'].replace('.bin', '.json')

    result = run_one_point_with_stream(p_frame_net, i_frame_net, args)

    result['ds_name'] = args['ds_name']
    result['seq'] = args['seq']
    result['rate_idx'] = args['rate_idx']
    result['qp_i'] = args['qp_i']
    result['qp_p'] = args['qp_p'] if 'qp_p' in args else args['qp_i']

    return result


def init_func(args, gpu_num):
    set_torch_env()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        if args.cuda_idx is not None:
            gpu_id = args.cuda_idx[gpu_id]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = "cuda:0"
    else:
        device = "cpu"

    global i_frame_net
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    i_frame_net.update(args.force_zero_thres)
    i_frame_net.half()

    global p_frame_net
    p_frame_net = DMC()
    if not args.force_intra:
        p_state_dict = get_state_dict(args.model_path_p)
        p_frame_net.load_state_dict(p_state_dict)
        p_frame_net = p_frame_net.to(device)
        p_frame_net.eval()
        p_frame_net.update(args.force_zero_thres)
        p_frame_net.half()


def main():
    begin_time = time.time()

    args = parse_args()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    if args.cuda_idx is not None:
        cuda_device = ','.join([str(s) for s in args.cuda_idx])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_num,
        initializer=init_func,
        initargs=(args, gpu_num)
    )
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    qp_i = []
    if args.qp_i is not None:
        assert len(args.qp_i) == rate_num
        qp_i = args.qp_i
    else:
        assert 2 <= rate_num <= DMC.get_qp_num()
        for i in np.linspace(0, DMC.get_qp_num() - 1, num=rate_num):
            qp_i.append(int(i + 0.5))

    if not args.force_intra:
        if args.qp_p is not None:
            assert len(args.qp_p) == rate_num
            qp_p = args.qp_p
        else:
            qp_p = qp_i

    print(f"[interlaced] testing {rate_num} rates, using qp: ", end='')
    for q in qp_i:
        print(f"{q}, ", end='')
    print()

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['qp_i'] = qp_i[rate_idx]
                if not args.force_intra:
                    cur_args['qp_p'] = qp_p[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['reset_interval'] = args.reset_interval
                cur_args['seq'] = seq
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq]['width']
                cur_args['intra_period'] = config[ds_name]['sequences'][seq]['intra_period']
                if args.force_intra:
                    cur_args['intra_period'] = 1
                if args.force_intra_period > 0:
                    cur_args['intra_period'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['calc_ssim'] = args.calc_ssim
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['check_existing'] = args.check_existing
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['verbose_json'] = args.verbose_json

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}

    for res in results:
        log_result[res['ds_name']][res['seq']][f"{res['rate_idx']:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()

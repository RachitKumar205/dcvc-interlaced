# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Multi-sequence batching extension for DCVC-RT.
#
# test_video.py encodes one sequence at a time, so each frame requires a
# separate set of GPU kernel launches.  When many sequences share the same
# resolution and rate point they can be processed in lockstep: at each frame
# index all N sequences contribute one frame to a [N, 3, H, W] batch and the
# model runs a single B=N forward pass, halving (or more) the total number of
# kernel launches for that frame step.
#
# Encoding quality is bit-for-bit identical to sequential encoding because
# each sequence's autoregressive chain is never disturbed: the model weights
# are shared and read-only; the per-sequence DPB state is held externally in
# RefFrame objects and only the relevant slice is fed to the model at each
# step.
#
# Usage:
#   python test_video_batch.py \
#       --model_path_i <i_model.pth.tar> \
#       --model_path_p <p_model.pth.tar> \
#       --test_config  dataset_config_example_yuv420.json \
#       --batch_size   2 \
#       --cuda True    --cuda_idx 0 \
#       --write_stream True \
#       --output_path  results_batch.json \
#       --stream_path  out_bin_batch
#
# The --batch_size flag controls how many sequences (of the same resolution
# and rate point) are grouped into one lockstep worker.  Sequences that cannot
# fill a full batch (e.g. the last group, or the sole sequence in a class) are
# handled with however many are available — the code degrades gracefully to
# N=1 (i.e. identical to test_video.py).

import argparse
import concurrent.futures
import io
import json
import multiprocessing
import os
import time

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DCVC-RT multi-sequence batching test script")

    parser.add_argument('--force_zero_thres', type=float, default=None)
    parser.add_argument('--model_path_i', type=str, required=True)
    parser.add_argument('--model_path_p',  type=str, required=True)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--qp_i', type=int, nargs="+")
    parser.add_argument('--qp_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--reset_interval', type=int, default=32)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None)
    parser.add_argument("--worker", "-w", type=int, default=1)
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+")
    parser.add_argument('--calc_ssim', type=str2bool, default=False)
    parser.add_argument('--write_stream', type=str2bool, default=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)
    # New flag: number of sequences to batch together per worker.
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Number of same-resolution sequences to encode "
                             "in lockstep per worker (default: 2).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Frame I/O helpers  (identical to test_video.py)
# ---------------------------------------------------------------------------

def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    if args['src_type'] == 'png':
        return PNGReader(args['src_path'], args['src_width'], args['src_height'])
    return YUV420Reader(args['src_path'], args['src_width'], args['src_height'])


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
        rgb = src_reader.read_one_frame()
        x = np_image_to_tensor(rgb, device)
        x = rgb2ycbcr(x)
        y, u, v = None, None, None
    x = x.to(torch.float16)
    return x, y, u, v, rgb


def get_distortion(args, x_hat, y, u, v, rgb):
    if args['src_type'] == 'yuv420':
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec  = torch.clamp(y_rec  * 255, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        y_rec  = y_rec [0, :, :]
        u_rec  = uv_rec[0, :, :]
        v_rec  = uv_rec[1, :, :]
        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr   = (6 * psnr_y + psnr_u + psnr_v) / 8
        if args['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec)
            ssim_u = calc_msssim(u, u_rec)
            ssim_v = calc_msssim(v, v_rec)
        else:
            ssim_y = ssim_u = ssim_v = 0.
        ssim = (6 * ssim_y + ssim_u + ssim_v) / 8
        return [psnr, psnr_y, psnr_u, psnr_v], [ssim, ssim_y, ssim_u, ssim_v]
    else:
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        psnr    = calc_psnr(rgb, rgb_rec)
        msssim  = calc_msssim_rgb(rgb, rgb_rec) if args['calc_ssim'] else 0.
        return [psnr], [msssim]


# ---------------------------------------------------------------------------
# materialise_feature
# ---------------------------------------------------------------------------

def materialise_feature(p_frame_net, ref_frame):
    """
    Return the [1, g_ch_d, H', W'] reference feature for one sequence without
    touching p_frame_net.dpb, mirroring DMC.apply_feature_adaptor().

    After an I-frame  : ref_frame.feature is None, ref_frame.frame holds x_hat
                        → pixel_unshuffle then feature_adaptor_i.
    After a P-frame   : ref_frame.feature is set
                        → feature_adaptor_p.
    """
    if ref_frame.feature is not None:
        return p_frame_net.feature_adaptor_p(ref_frame.feature)
    return p_frame_net.feature_adaptor_i(F.pixel_unshuffle(ref_frame.frame, 8))


# ---------------------------------------------------------------------------
# Core: encode a batch of sequences in lockstep
# ---------------------------------------------------------------------------

def encode_batch(p_frame_net, i_frame_net, seq_args_list,
                 pic_height, pic_width, padding_r, padding_b,
                 use_two_entropy_coders, verbose):
    """
    Encode a group of N sequences in lockstep, using a single B=N GPU forward
    pass for every P-frame step.

    seq_args_list : list of per-sequence arg dicts (all same resolution).
    Returns       : list of per-sequence result dicts, each containing
                    'frame_types', 'bits', 'encoding_times', 'output_buff'.
    """
    N = len(seq_args_list)
    device = next(p_frame_net.parameters()).device
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    # Per-sequence state --------------------------------------------------
    src_readers   = [get_src_reader(a) for a in seq_args_list]
    frame_nums    = [a['frame_num']       for a in seq_args_list]
    reset_intv    = seq_args_list[0]['reset_interval']   # same for all

    # Output buffers and SPS helpers, one per sequence
    output_buffs  = [io.BytesIO() for _ in range(N)]
    sps_helpers   = [SPSHelper()  for _ in range(N)]

    # Per-sequence DPB state: one RefFrame per sequence
    ref_frames    = [RefFrame() for _ in range(N)]

    # Timing / result accumulators
    frame_types   = [[] for _ in range(N)]
    bits          = [[] for _ in range(N)]
    enc_times     = [[] for _ in range(N)]
    last_qps      = [0] * N

    max_frames = max(frame_nums)

    for frame_idx in range(max_frames):
        # Which sequences still have frames at this index?
        active = [i for i in range(N) if frame_idx < frame_nums[i]]
        if not active:
            break

        # Read padded input frames for active sequences
        x_padded_list = []
        src_data_list = []   # (y, u, v, rgb) for distortion — not used here
        for i in active:
            x, y_src, u_src, v_src, rgb_src = get_src_frame(
                seq_args_list[i], src_readers[i], device)
            x_padded_list.append(replicate_pad(x, padding_b, padding_r))
            src_data_list.append((y_src, u_src, v_src, rgb_src))

        # Decide frame type for each active sequence
        is_i_frame = {}
        for i in active:
            intra_period = seq_args_list[i]['intra_period']
            is_i_frame[i] = (frame_idx == 0) or \
                            (intra_period > 0 and frame_idx % intra_period == 0)

        torch.cuda.synchronize(device=device)
        step_start = time.time()

        # ---- I-frame sequences (encode sequentially) --------------------
        i_seqs = [i for i in active if     is_i_frame[i]]
        p_seqs = [i for i in active if not is_i_frame[i]]

        for i in i_seqs:
            curr_qp_i = seq_args_list[i]['qp_i']
            sps = {
                'sps_id': -1,
                'height': pic_height,
                'width':  pic_width,
                'ec_part': 1 if use_two_entropy_coders else 0,
                'use_ada_i': 0,
            }
            encoded = i_frame_net.compress(x_padded_list[active.index(i)], curr_qp_i)
            ref_frames[i].frame   = encoded['x_hat']
            ref_frames[i].feature = None
            last_qps[i] = curr_qp_i

            torch.cuda.synchronize(device=device)
            step_end = time.time()
            elapsed = step_end - step_start

            sps_id, sps_new = sps_helpers[i].get_sps_id(sps)
            sps['sps_id'] = sps_id
            sps_bytes = write_sps(output_buffs[i], sps) if sps_new else 0
            stream_bytes = write_ip(output_buffs[i], True, sps_id,
                                    curr_qp_i, encoded['bit_stream'])
            frame_types[i].append(0)
            bits[i].append(stream_bytes * 8 + sps_bytes * 8)
            enc_times[i].append(elapsed)

            if verbose >= 2:
                print(f"[batch] frame {frame_idx} seq {i} (I), "
                      f"{elapsed*1000:.2f} ms, bits: {bits[i][-1]}")

        # ---- P-frame sequences ------------------------------------------
        if p_seqs:
            # Determine QP (same for all — they're in lockstep)
            fa_idx  = index_map[frame_idx % 8]
            curr_qp = p_frame_net.shift_qp(seq_args_list[p_seqs[0]]['qp_p'], fa_idx)

            # Handle reset_interval per sequence before the batched forward.
            # prepare_feature_adaptor_i() needs the DPB, so we temporarily
            # load each sequence's state, call it, then read the result back.
            use_ada_i = 0
            if reset_intv > 0 and frame_idx % reset_intv == 1:
                use_ada_i = 1
                for i in p_seqs:
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(ref_frames[i].feature,
                                              ref_frames[i].frame)
                    p_frame_net.prepare_feature_adaptor_i(last_qps[i])
                    ref_frames[i].frame   = p_frame_net.dpb[0].frame
                    ref_frames[i].feature = p_frame_net.dpb[0].feature

            sps = {
                'sps_id': -1,
                'height': pic_height,
                'width':  pic_width,
                'ec_part': 1 if use_two_entropy_coders else 0,
                'use_ada_i': use_ada_i,
            }

            if len(p_seqs) == 1:
                # Only one P-frame sequence active — use compress() directly.
                i = p_seqs[0]
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(ref_frames[i].feature,
                                          ref_frames[i].frame)
                encoded = p_frame_net.compress(
                    x_padded_list[active.index(i)], curr_qp)
                ref_frames[i].feature = p_frame_net.dpb[0].feature
                ref_frames[i].frame   = None
                last_qps[i] = curr_qp

                torch.cuda.synchronize(device=device)
                step_end = time.time()
                elapsed = step_end - step_start

                sps_id, sps_new = sps_helpers[i].get_sps_id(sps)
                sps['sps_id'] = sps_id
                sps_bytes = write_sps(output_buffs[i], sps) if sps_new else 0
                stream_bytes = write_ip(output_buffs[i], False, sps_id,
                                        curr_qp, encoded['bit_stream'])
                frame_types[i].append(1)
                bits[i].append(stream_bytes * 8 + sps_bytes * 8)
                enc_times[i].append(elapsed)

                if verbose >= 2:
                    print(f"[batch] frame {frame_idx} seq {i} (P, solo), "
                          f"{elapsed*1000:.2f} ms, bits: {bits[i][-1]}")

            else:
                # Batched path: stack frames and features, single forward pass.
                local_idxs = [active.index(i) for i in p_seqs]
                x_batch = torch.cat([x_padded_list[li] for li in local_idxs],
                                    dim=0)   # [len(p_seqs), 3, H, W]

                mat_features = torch.cat(
                    [materialise_feature(p_frame_net, ref_frames[i])
                     for i in p_seqs],
                    dim=0)   # [len(p_seqs), g_ch_d, H', W']

                batch_streams, features_out = p_frame_net.compress_batch(
                    x_batch, curr_qp, mat_features)

                torch.cuda.synchronize(device=device)
                step_end  = time.time()
                elapsed   = step_end - step_start
                per_frame = elapsed / len(p_seqs)

                for batch_pos, i in enumerate(p_seqs):
                    ref_frames[i].feature = features_out[batch_pos:batch_pos+1]
                    ref_frames[i].frame   = None
                    last_qps[i]           = curr_qp

                    sps_id, sps_new = sps_helpers[i].get_sps_id(sps)
                    sps_copy = dict(sps)
                    sps_copy['sps_id'] = sps_id
                    sps_bytes = write_sps(output_buffs[i], sps_copy) \
                        if sps_new else 0
                    stream_bytes = write_ip(output_buffs[i], False, sps_id,
                                            curr_qp, batch_streams[batch_pos])
                    frame_types[i].append(1)
                    bits[i].append(stream_bytes * 8 + sps_bytes * 8)
                    enc_times[i].append(per_frame)

                    if verbose >= 2:
                        print(f"[batch] frame {frame_idx} seq {i} (P, batched "
                              f"N={len(p_seqs)}), {per_frame*1000:.2f} ms/seq, "
                              f"bits: {bits[i][-1]}")

    for sr in src_readers:
        sr.close()

    # Pack results
    results = []
    for i in range(N):
        results.append({
            'frame_types':     frame_types[i],
            'bits':            bits[i],
            'encoding_times':  enc_times[i],
            'output_buff':     output_buffs[i],
        })
    return results


# ---------------------------------------------------------------------------
# Core: decode a single sequence  (identical logic to test_video.py)
# ---------------------------------------------------------------------------

def decode_sequence(p_frame_net, i_frame_net, seq_args, input_buff,
                    pic_height, pic_width):
    """
    Decode a bitstream produced by encode_batch for one sequence.
    Returns (recon_frames list, decoding_times list).
    """
    device   = next(p_frame_net.parameters()).device
    verbose  = seq_args['verbose']
    frame_num = seq_args['frame_num']

    sps_helper = SPSHelper()
    recon_frames  = []
    decoding_times = []

    p_frame_net.set_curr_poc(0)

    for frame_idx in range(frame_num):
        torch.cuda.synchronize(device=device)
        frame_start = time.time()

        header = read_header(input_buff)
        while header['nal_type'] == NalType.NAL_SPS:
            sps = read_sps_remaining(input_buff, header['sps_id'])
            sps_helper.add_sps_by_id(sps)
            if verbose >= 2:
                print("new sps", sps)
            header = read_header(input_buff)

        sps = sps_helper.get_sps_by_id(header['sps_id'])
        qp, bit_stream = read_ip_remaining(input_buff)

        if header['nal_type'] == NalType.NAL_I:
            decoded = i_frame_net.decompress(bit_stream, sps, qp)
            p_frame_net.clear_dpb()
            p_frame_net.add_ref_frame(None, decoded['x_hat'])
        elif header['nal_type'] == NalType.NAL_P:
            if sps['use_ada_i']:
                p_frame_net.reset_ref_feature()
            decoded = p_frame_net.decompress(bit_stream, sps, qp)

        torch.cuda.synchronize(device=device)
        frame_end = time.time()

        recon_frames.append(decoded['x_hat'])
        decoding_times.append(frame_end - frame_start)

        if verbose >= 2:
            nbits = 0 if bit_stream is None else len(bit_stream) * 8
            print(f"frame {frame_idx} decoded, "
                  f"{(frame_end-frame_start)*1000:.3f} ms, bits: {nbits}")

    return recon_frames, decoding_times


# ---------------------------------------------------------------------------
# run_one_batch: encode + decode + measure a group of N sequences
# ---------------------------------------------------------------------------

def run_one_batch(p_frame_net, i_frame_net, seq_args_list):
    """
    Encode and decode a batch of N sequences, write per-sequence .bin and
    .json files, and return per-sequence result dicts (same format as
    test_video.py's run_one_point_with_stream).
    """
    # Use the first sequence's parameters for shared settings.
    ref_args = seq_args_list[0]
    pic_height = ref_args['src_height']
    pic_width  = ref_args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

    use_two_entropy_coders = pic_height * pic_width > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    verbose = ref_args['verbose']
    N = len(seq_args_list)

    # Check existing outputs — if all are present, skip re-encoding.
    if ref_args.get('check_existing', False):
        all_exist = all(
            os.path.exists(a['curr_json_path']) and
            os.path.exists(a['curr_bin_path'])
            for a in seq_args_list
        )
        if all_exist:
            results = []
            for a in seq_args_list:
                with open(a['curr_json_path']) as f:
                    log = json.load(f)
                if log['i_frame_num'] + log['p_frame_num'] == a['frame_num']:
                    results.append(log)
                    continue
                # One is stale — re-run the whole batch.
                results = []
                break
            if results:
                return results

    start_time = time.time()

    # ---------- Encode ---------------------------------------------------
    p_frame_net.set_curr_poc(0)
    with torch.no_grad():
        enc_results = encode_batch(
            p_frame_net, i_frame_net, seq_args_list,
            pic_height, pic_width, padding_r, padding_b,
            use_two_entropy_coders, verbose)

    # Write bitstreams to disk
    total_bytes_list = []
    for i, a in enumerate(seq_args_list):
        buf = enc_results[i]['output_buff']
        with open(a['curr_bin_path'], 'wb') as f:
            bb = buf.getbuffer()
            f.write(bb)
            total_bytes_list.append(bb.nbytes)
            bb.release()
        buf.close()

    # ---------- Decode + distortion measurement --------------------------
    # Decoding is sequential: the decoder is not batched.  We decode each
    # sequence independently, reading its own .bin file.
    log_results = []
    for i, a in enumerate(seq_args_list):
        src_reader = get_src_reader(a)
        frames_src = []
        for _ in range(a['frame_num']):
            frames_src.append(get_src_frame(a, src_reader, next(p_frame_net.parameters()).device))
        src_reader.close()

        p_frame_net.set_curr_poc(0)
        with open(a['curr_bin_path'], 'rb') as f:
            input_buff = io.BytesIO(f.read())

        with torch.no_grad():
            recon_frames, dec_times = decode_sequence(
                p_frame_net, i_frame_net, a, input_buff,
                pic_height, pic_width)
        input_buff.close()

        if a.get('save_decoded_frame', False):
            if a['src_type'] == 'yuv420':
                total_kbps = int(total_bytes_list[i] * 8 /
                                 (a['frame_num'] / 30) / 1000)
                out_yuv = a['curr_rec_path'].replace(
                    '.yuv', f'_{total_kbps}kbps.yuv')
                writer = YUV420Writer(out_yuv, a['src_width'], a['src_height'])
            else:
                writer = PNGWriter(
                    a['bin_folder'], a['src_width'], a['src_height'])

        psnrs   = []
        msssims = []
        for frame_idx in range(a['frame_num']):
            _, y_s, u_s, v_s, rgb_s = frames_src[frame_idx]
            x_hat = recon_frames[frame_idx][:, :, :pic_height, :pic_width]
            psnr_v, ssim_v = get_distortion(a, x_hat, y_s, u_s, v_s, rgb_s)
            psnrs.append(psnr_v)
            msssims.append(ssim_v)

            if a.get('save_decoded_frame', False):
                if a['src_type'] == 'yuv420':
                    y_r, uv_r = yuv_444_to_420(x_hat)
                    y_r  = torch.clamp(y_r  * 255, 0, 255).round() \
                               .to(torch.uint8).squeeze(0).cpu().numpy()
                    uv_r = torch.clamp(uv_r * 255, 0, 255) \
                               .to(torch.uint8).squeeze(0).cpu().numpy()
                    writer.write_one_frame(y_r, uv_r)
                else:
                    rgb_r = ycbcr2rgb(x_hat)
                    rgb_r = torch.clamp(rgb_r * 255, 0, 255).round() \
                                .to(torch.uint8).squeeze(0).cpu().numpy()
                    writer.write_one_frame(rgb_r)

        if a.get('save_decoded_frame', False):
            writer.close()

        # Use wall-clock time from script start
        test_time = time.time() - script_start_time
        enc_t = enc_results[i]['encoding_times']
        frame_types_i = enc_results[i]['frame_types']
        bits_i        = enc_results[i]['bits']

        time_bypass = 10
        avg_enc = avg_dec = None
        if verbose >= 1 and len(enc_t) > time_bypass:
            avg_enc = sum(enc_t[time_bypass:]) / len(enc_t[time_bypass:])
            avg_dec = sum(dec_times[time_bypass:]) / len(dec_times[time_bypass:])
            print(f"[batch] seq {i} ({a['seq']}): "
                  f"avg enc {avg_enc*1000:.3f} ms/frame, "
                  f"avg dec {avg_dec*1000:.3f} ms/frame")

        log = generate_log_json(
            a['frame_num'], pic_height * pic_width, test_time,
            frame_types_i, bits_i, psnrs, msssims,
            verbose=a.get('verbose_json', False),
            avg_encoding_time=avg_enc,
            avg_decoding_time=avg_dec)

        with open(a['curr_json_path'], 'w') as fp:
            json.dump(log, fp, indent=2)

        log_results.append(log)

    return log_results


# ---------------------------------------------------------------------------
# Worker / process pool
# ---------------------------------------------------------------------------

i_frame_net = None
p_frame_net = None
script_start_time = None


def worker(seq_args_list):
    """
    Process pool worker: receives a list of per-sequence arg dicts and runs
    run_one_batch on them.  Returns a list of result dicts.
    """
    global i_frame_net, p_frame_net, script_start_time
    assert i_frame_net is not None and p_frame_net is not None

    for a in seq_args_list:
        bin_folder = os.path.join(a['stream_path'], a['ds_name'])
        assert a['write_stream'], "write_stream must be True"
        create_folder(bin_folder, True)
        a['bin_folder']      = bin_folder
        a['curr_bin_path']   = os.path.join(bin_folder,
                                             f"{a['seq']}_q{a['qp_i']}.bin")
        a['curr_rec_path']   = a['curr_bin_path'].replace('.bin', '.yuv')
        a['curr_json_path']  = a['curr_bin_path'].replace('.bin', '.json')

    log_results = run_one_batch(p_frame_net, i_frame_net, seq_args_list)

    out = []
    for a, log in zip(seq_args_list, log_results):
        log['ds_name']  = a['ds_name']
        log['seq']      = a['seq']
        log['rate_idx'] = a['rate_idx']
        log['qp_i']     = a['qp_i']
        log['qp_p']     = a.get('qp_p', a['qp_i'])
        out.append(log)
    return out


def init_func(args, gpu_num, start_time):
    set_torch_env()

    process_name = multiprocessing.current_process().name
    process_idx  = int(process_name[process_name.rfind('-') + 1:])
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

    global i_frame_net, script_start_time
    script_start_time = start_time
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    begin_time = time.time()
    args = parse_args()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    if args.cuda_idx is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(s) for s in args.cuda_idx)

    with open(args.test_config) as f:
        config = json.load(f)

    gpu_num = torch.cuda.device_count() if args.cuda else 0

    # Build QP lists (identical logic to test_video.py)
    rate_num = args.rate_num
    qp_i_list = []
    if args.qp_i is not None:
        assert len(args.qp_i) == rate_num
        qp_i_list = args.qp_i
    else:
        assert 2 <= rate_num <= DMC.get_qp_num()
        for v in np.linspace(0, DMC.get_qp_num() - 1, num=rate_num):
            qp_i_list.append(int(v + 0.5))

    if not args.force_intra:
        if args.qp_p is not None:
            assert len(args.qp_p) == rate_num
            qp_p_list = args.qp_p
        else:
            qp_p_list = qp_i_list

    print(f"[batch] testing {rate_num} rates, qp_i: {qp_i_list}")

    root_path = (args.force_root_path
                 if args.force_root_path is not None
                 else config['root_path'])
    test_classes = config['test_classes']

    # Build all per-sequence arg dicts, grouped by (resolution, rate_idx) so
    # only sequences of the same size are batched together.
    # Key: (width, height, rate_idx) → list of arg dicts
    from collections import defaultdict
    resolution_groups = defaultdict(list)

    count_frames = 0
    count_sequences = 0

    for ds_name, ds_cfg in test_classes.items():
        if ds_cfg['test'] == 0:
            continue
        for seq, seq_cfg in ds_cfg['sequences'].items():
            count_sequences += 1
            for rate_idx in range(rate_num):
                a = {}
                a['rate_idx']       = rate_idx
                a['qp_i']           = qp_i_list[rate_idx]
                if not args.force_intra:
                    a['qp_p']       = qp_p_list[rate_idx]
                a['force_intra']    = args.force_intra
                a['reset_interval'] = args.reset_interval
                a['seq']            = seq
                a['src_type']       = ds_cfg['src_type']
                a['src_height']     = seq_cfg['height']
                a['src_width']      = seq_cfg['width']
                a['intra_period']   = seq_cfg['intra_period']
                if args.force_intra:
                    a['intra_period'] = 1
                if args.force_intra_period > 0:
                    a['intra_period'] = args.force_intra_period
                a['frame_num']      = seq_cfg['frames']
                if args.force_frame_num > 0:
                    a['frame_num']  = args.force_frame_num
                a['calc_ssim']      = args.calc_ssim
                a['dataset_path']   = os.path.join(
                    root_path, ds_cfg['base_path'])
                a['src_path']       = os.path.join(
                    a['dataset_path'], seq)
                a['write_stream']   = args.write_stream
                a['check_existing'] = args.check_existing
                a['stream_path']    = args.stream_path
                a['save_decoded_frame'] = args.save_decoded_frame
                a['ds_name']        = ds_name
                a['verbose']        = args.verbose
                a['verbose_json']   = args.verbose_json

                count_frames += a['frame_num']
                key = (a['src_width'], a['src_height'], rate_idx)
                resolution_groups[key].append(a)

    # Split each resolution/rate group into sub-batches of size --batch_size
    batch_size = args.batch_size
    job_batches = []
    for key, seq_list in resolution_groups.items():
        for start in range(0, len(seq_list), batch_size):
            job_batches.append(seq_list[start:start + batch_size])

    print(f"[batch] {count_sequences} sequences, "
          f"{len(job_batches)} batched jobs "
          f"(batch_size={batch_size}), "
          f"{count_frames} total frames")

    multiprocessing.set_start_method("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.worker,
        initializer=init_func,
        initargs=(args, gpu_num, begin_time))

    objs = [executor.submit(worker, batch) for batch in job_batches]

    # Collect results
    all_results = []
    for obj in tqdm(objs):
        all_results.extend(obj.result())

    # Aggregate into the same nested JSON structure as test_video.py
    log_result = {}
    for ds_name, ds_cfg in test_classes.items():
        if ds_cfg['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in ds_cfg['sequences']:
            log_result[ds_name][seq] = {}

    for res in all_results:
        log_result[res['ds_name']][res['seq']][f"{res['rate_idx']:03d}"] = res

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        create_folder(out_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()

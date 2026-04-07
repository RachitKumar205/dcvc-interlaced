#!/usr/bin/env python3
"""
setup_360p_test.py

Complete setup script for 360p batching experiment.
Creates downsampled YUV files, symlinks, and dataset config.

Usage:
    python setup_360p_test.py
"""

import numpy as np
import json
import os
import sys

def crop_yuv_420(in_path, out_path, w_in, h_in, w_out, h_out, num_frames):
    """Crop YUV 4:2:0 file to smaller resolution."""
    print(f"Cropping {in_path}")
    print(f"  Input:  {w_in}x{h_in}")
    print(f"  Output: {w_out}x{h_out}")
    print(f"  Frames: {num_frames}")
    
    y_in_size = w_in * h_in
    uv_in_size = (w_in // 2) * (h_in // 2)
    
    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        for i in range(num_frames):
            if (i + 1) % 10 == 0:
                print(f"  Processing frame {i+1}/{num_frames}...")
            
            # Read YUV420 planes
            y = np.frombuffer(fin.read(y_in_size), dtype=np.uint8).reshape(h_in, w_in)
            u = np.frombuffer(fin.read(uv_in_size), dtype=np.uint8).reshape(h_in // 2, w_in // 2)
            v = np.frombuffer(fin.read(uv_in_size), dtype=np.uint8).reshape(h_in // 2, w_in // 2)
            
            # Crop to output size
            y_out = y[:h_out, :w_out]
            u_out = u[:h_out//2, :w_out//2]
            v_out = v[:h_out//2, :w_out//2]
            
            # Write cropped planes
            fout.write(y_out.tobytes())
            fout.write(u_out.tobytes())
            fout.write(v_out.tobytes())
    
    print(f"Created {out_path}\n")

def create_symlink(target, link_name):
    """Create symlink (cross-platform)."""
    if os.path.exists(link_name):
        os.remove(link_name)
    
    # On Windows, use copy if symlink fails
    try:
        os.symlink(target, link_name)
        print(f"Created symlink: {link_name} -> {target}")
    except OSError:
        import shutil
        shutil.copy2(target, link_name)
        print(f"Created copy (symlink failed): {link_name}")

def create_config(output_path, width, height, frames):
    """Create dataset config JSON for 360p batch test."""
    config = {
        "root_path": "test_data/",
        "test_classes": {
            "UVG": {
                "base_path": "UVG",
                "src_type": "yuv420",
                "test": 1,
                "sequences": {
                    f"ShakeNDry_{width}x{height}_120fps_420_8bit_YUV.yuv": {
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "intra_period": -1
                    },
                    f"ShakeNDry_{width}x{height}_120fps_420_8bit_YUV_copy.yuv": {
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "intra_period": -1
                    }
                }
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created config: {output_path}\n")

def main():
    # Configuration
    input_yuv = 'test_data/UVG/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv'
    output_yuv = 'test_data/UVG/ShakeNDry_640x360_120fps_420_8bit_YUV.yuv'
    output_copy = 'test_data/UVG/ShakeNDry_640x360_120fps_420_8bit_YUV_copy.yuv'
    config_output = 'dataset_config_batch2_360p.json'
    
    w_in, h_in = 1920, 1080
    w_out, h_out = 640, 360
    num_frames = 100
    
    # Check input exists
    if not os.path.exists(input_yuv):
        print(f"ERROR: Input file not found: {input_yuv}")
        sys.exit(1)
    
    print("=" * 70)
    print("DCVC-RT 360p Test Setup")
    print("=" * 70)
    print()
    
    # Step 1: Crop YUV
    if not os.path.exists(output_yuv):
        crop_yuv_420(input_yuv, output_yuv, w_in, h_in, w_out, h_out, num_frames)
    else:
        print(f"Skipping crop (already exists): {output_yuv}\n")
    
    # Step 2: Create symlink/copy
    create_symlink(os.path.basename(output_yuv), output_copy)
    print()
    
    # Step 3: Create config
    create_config(config_output, w_out, h_out, num_frames)
    
    print("=" * 70)
    print("Setup complete! Run the batched test with:")
    print()
    print(f"python test_video_batch.py \\")
    print(f"  --model_path_i ./checkpoints/cvpr2025_image.pth.tar \\")
    print(f"  --model_path_p ./checkpoints/cvpr2025_video.pth.tar \\")
    print(f"  --rate_num 4 --test_config ./{config_output} \\")
    print(f"  --cuda 1 -w 1 --write_stream 1 --force_zero_thres 0.12 \\")
    print(f"  --force_intra_period -1 --reset_interval 64 \\")
    print(f"  --force_frame_num 100 --output_path output_360p.json --verbose 1")
    print("=" * 70)

if __name__ == "__main__":
    main()
